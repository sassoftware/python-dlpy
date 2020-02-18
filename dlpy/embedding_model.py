#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' The Embedding Model class adds training, evaluation,feature analysis routines for learning embedding '''

from copy import deepcopy

from dlpy import Model
from dlpy.layers import Input, EmbeddingLoss, OutputLayer, Keypoints
from .attribute_utils import create_extended_attributes
from .image_embedding import ImageEmbeddingTable
from .model import DataSpec
from .network import WeightsTable
from dlpy.utils import DLPyError


class EmbeddingModel(Model):
    input_layer_name_prefix = 'InputLayer_'
    embedding_layer_name_prefix = 'EmbeddingLayer_'
    embedding_loss_layer_name = 'EmbeddingLossLayer'
    number_of_branches = 0
    data_specs = None
    embedding_model_type = None
    branch_input_tensor = None
    branch_output_tensor = None

    @classmethod
    def build_embedding_model(cls, branch, model_table=None, embedding_model_type='Siamese',
                              embedding_layer=None, margin=None):
        '''

        Build an embedding model based on a given model branch and model type

        Parameters
        ----------
        branch : Model
            Specifies the base model that is used as branches for embedding model.
        model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        embedding_model_type : string, optional
            Specifies the embedding model type that the created table will be applied for training.
            Valid values: Siamese, Triplet, and Quartet.
            Default: Siamese
        embedding_layer: Layer, optional
            Specifies a dense layer as the embedding layer. For instance, Dense(n=10, act='identity') defines
            the embedding dimension is 10. When it is not given, the last layer (except the task layers)
            in the branch model will be used as the embedding layer.
        margin: double, optional
            Specifies the margin value used by the embedding model. When it is not given, for Siamese, margin is 2.0.
            Otherwise, margin is 0.0.

        Returns
        -------
        :class:`Model`

        '''

        # check the branch type
        if not isinstance(branch, Model):
            raise DLPyError('The branch option must contain a valid model')

        # the branch must be built using functional APIs
        # only functional model has the attr output_layers
        if not hasattr(branch, 'output_layers'):
            print("NOTE: Convert the branch model into a functional model.")
            branch_tensor = branch.to_functional_model()
        else:
            branch_tensor = deepcopy(branch)

        # always reset this local tensor to 0
        branch_tensor.number_of_instances = 0

        # the branch cannot contain other task layers
        if len(branch_tensor.output_layers) != 1:
            raise DLPyError('The branch model cannot contain more than one output layer')
        elif branch_tensor.output_layers[0].type == OutputLayer.type or \
                branch_tensor.output_layers[0].type == Keypoints.type:
            print("NOTE: Remove the task layers from the model.")
            branch_tensor.layers.remove(branch_tensor.output_layers[0])
            branch_tensor.output_layers[0] = branch_tensor.layers[-1]
        elif branch_tensor.output_layers[0].can_be_last_layer:
            raise DLPyError('The branch model cannot contain task layer except output or keypoints layer.')

        # check embedding_model_type
        if embedding_model_type.lower() not in ['siamese', 'triplet', 'quartet']:
            raise DLPyError('Only Siamese, Triplet, and Quartet are valid.')

        if embedding_model_type.lower() == 'siamese':
            if margin is None:
                margin = 2.0
            cls.number_of_branches = 2
        elif embedding_model_type.lower() == 'triplet':
            if margin is None:
                margin = 0.0
            cls.number_of_branches = 3
        elif embedding_model_type.lower() == 'quartet':
            if margin is None:
                margin = 0.0
            cls.number_of_branches = 4

        cls.embedding_model_type = embedding_model_type

        # build the branches
        input_layers = []
        branch_layers = []
        for i_branch in range(cls.number_of_branches):
            temp_input_layer = Input(**branch_tensor.layers[0].config, name=cls.input_layer_name_prefix + str(i_branch))
            temp_branch = branch_tensor(temp_input_layer)  # return a list of tensors
            if embedding_layer:
                temp_embed_layer = deepcopy(embedding_layer)
                temp_embed_layer.name = cls.embedding_layer_name_prefix + str(i_branch)
                temp_branch = temp_embed_layer(temp_branch)
                # change tensor to a list
                temp_branch = [temp_branch]
            else:
                # change the last layer name to the embedding layer name
                temp_branch[-1]._op.name = cls.embedding_layer_name_prefix + str(i_branch)

            if i_branch == 0:
                cls.branch_input_tensor = temp_input_layer
                if len(temp_branch) == 1:
                    cls.branch_output_tensor = temp_branch[0]
                else:
                    cls.branch_output_tensor = temp_branch

            # append these layers to the current branch
            input_layers.append(temp_input_layer)
            branch_layers = branch_layers + temp_branch

        # add the embedding loss layer
        loss_layer = EmbeddingLoss(margin=margin, name=cls.embedding_loss_layer_name)(branch_layers)

        # create the model DAG using all the above model information
        model = EmbeddingModel(branch.conn, model_table=model_table, inputs=input_layers, outputs=loss_layer)

        # sharing weights
        # get all layer names from one branch
        num_l = int((len(model.layers) - 1) / cls.number_of_branches)
        br1_name = [i.name for i in model.layers[:num_l - 1]]

        # build the list that contain the shared layers
        share_list = []
        n_id = 0
        n_to = n_id + cls.number_of_branches
        for l in br1_name[1:]:
            share_list.append({l: [l + '_' + str(i + 1) for i in range(n_id + 1, n_to)]})

        # add embedding layers
        share_list.append({cls.embedding_layer_name_prefix + str(0):
                               [cls.embedding_layer_name_prefix + str(i)
                                for i in range(1, cls.number_of_branches)]})

        model.share_weights(share_list)

        model.compile()

        # generate data_specs
        if embedding_model_type.lower() == 'siamese':
            cls.data_specs = [DataSpec(type_='image', layer=cls.input_layer_name_prefix + '0', data=['_image_']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '1', data=['_image_1']),
                              DataSpec(type_='numnom', layer=cls.embedding_loss_layer_name, data=['_dissimilar_'])]
        elif embedding_model_type.lower() == 'triplet':
            cls.data_specs = [DataSpec(type_='image', layer=cls.input_layer_name_prefix + '0', data=['_image_']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '1', data=['_image_1']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '2', data=['_image_2'])]

        elif embedding_model_type.lower() == 'quartet':
            cls.data_specs = [DataSpec(type_='image', layer=cls.input_layer_name_prefix + '0', data=['_image_']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '1', data=['_image_1']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '2', data=['_image_2']),
                              DataSpec(type_='image', layer=cls.input_layer_name_prefix + '3', data=['_image_3'])]

        return model

    def fit_embedding_model(self, optimizer,
                            data=None, path=None, n_samples=512, label_level=-2,
                            resize_width=None, resize_height=None,
                            max_iter=1,
                            valid_table=None, valid_freq=1, gpu=None, seed=0, record_seed=0,
                            save_best_weights=False, n_threads=None,
                            train_from_scratch=None):

        """
        Fitting a deep learning model for embedding learning.

        Parameters
        ----------

        optimizer : :class:`Optimizer`
            Specifies the parameters for the optimizer.
        data : class:`ImageEmbeddingTable`, optional
            This is the input data. It muse be a ImageEmbeddingTable object. Either data or path has to be specified.
        path : string, optional
            The path to the image directory on the server.
            Path may be absolute, or relative to the current caslib root.
            when path is specified, the data option will be ignored.
            A new sample of data will be randomly generated after the number of epochs defined in Optimizer.
            max_iter defines how many iterations the random sample will be generated.
        n_samples : int, optional
            Number of samples to generate.
            Default: 512
        label_level : int, optional
            Specifies which path level should be used to generate the class labels for each image.
            This class label determines whether a given image pair belongs to the same class.
            For instance, label_level = 1 means the first directory and label_level = -2 means the last directory.
            This internally use the SAS scan function
            (check https://www.sascrunch.com/scan-function.html for more details).
            Default: -2
        resize_width : int, optional
            Specifies the image width that needs be resized to. When resize_width is not given, it will be reset to
            the specified resize_height.
        resize_height : int, optional
            Specifies the image height that needs be resized to. When resize_height is not given, it will be reset to
            the specified resize_width.
        max_iter : int, optional
            Hard limit on iterations when randomly generating data.
            Default: 1
        valid_table : string or CASTable, optional
            Specifies the table with the validation data. The validation
            table must have the same columns and data types as the training table.
        valid_freq : int, optional
            Specifies the frequency for scoring the validation table.
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing unit hardware.
            The simplest way to use GPU processing is to specify "gpu=1".
            In this case, the default values of other GPU parameters are used.
            Setting gpu=1 enables all available GPU devices for use. Setting
            gpu=0 disables GPU processing.
        seed : double, optional
            specifies the random number seed for the random number generator
            in SGD. The default value, 0, and negative values indicate to use
            random number streams based on the computer clock. Specify a value
            that is greater than 0 for a reproducible random number sequence.
        record_seed : double, optional
            specifies the random number seed for the random record selection
            within a worker. The default value 0 disables random record selection.
            Records are read as they are laid out in memory.
            Negative values indicate to use random number streams based on the
            computer clock.
        save_best_weights : bool, optional
            When set to True, it keeps the weights that provide the smallest
            loss error.
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.
        train_from_scratch : bool, optional
            When set to True, it ignores the existing weights and trains the model from the scracth.

        Returns
        --------
        :class:`CASResults` or a list of `CASResults` when the path option is specified

        """

        # check options
        if data is None and path is None:
            raise DLPyError('Either the data option or path must be specified to generate the input data')

        if data is not None and path is not None:
            print('Note: the data option will be ignored and the path option will be used to generate the input '
                  'data')

        # check the data type
        if path is None:
            if not isinstance(data, ImageEmbeddingTable):
                raise DLPyError('The data option must contain a valid embedding table')
            if data.embedding_model_type.lower() != self.embedding_model_type:
                raise DLPyError('The data option must contain a valid embedding table for ' + self.embedding_model_type)

        # use the data option to train a model
        if path is None:
            res = self.fit(data, inputs=None, target=None, data_specs=self.data_specs,
                           optimizer=optimizer,
                           valid_table=valid_table, valid_freq=valid_freq, gpu=gpu,
                           seed=seed, record_seed=record_seed,
                           force_equal_padding=True,
                           save_best_weights=save_best_weights, n_threads=n_threads,
                           target_order=None, train_from_scratch=train_from_scratch)
        else:  # use the path option to generate the input data
            import time
            res = []
            time_start = time.time()
            for data_iter in range(0, max_iter):
                # generate a new data table
                time_0 = time.time()
                data = ImageEmbeddingTable.load_files(self.conn, path=path, n_samples=n_samples,
                                                      label_level=label_level,
                                                      embedding_model_type=self.embedding_model_type,
                                                      resize_width=resize_width, resize_height=resize_height)
                print('Note: data generation took {} (s) at iteration {}'.format(time.time() - time_0, data_iter))

                # train the model using this data
                if data_iter == 0:
                    train_from_scratch_real = train_from_scratch
                else:
                    train_from_scratch_real = False
                res_t = self.fit(data, inputs=None, target=None, data_specs=self.data_specs,
                                 optimizer=optimizer,
                                 valid_table=valid_table, valid_freq=valid_freq, gpu=gpu,
                                 seed=seed, record_seed=record_seed,
                                 force_equal_padding=True,
                                 save_best_weights=save_best_weights, n_threads=n_threads,
                                 target_order=None, train_from_scratch=train_from_scratch_real)
                res.append(res_t)
                # drop this data
                data.droptable()
            print('Note: Training with data generation took {} (s)'.format(time.time() - time_start))
        return res

    def deploy_embedding_model(self, path, output_format='astore', model_type='branch'):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the location to store the model files.
            If the output_format is set to castable, then the location has to be on the server-side.
            Otherwise, the location has to be on the client-side.
        output_format : string, optional
            Specifies the format of the deployed model.
            When astore is specified, the learned embedding features will be output as well.
            Valid Values: astore, castable, or onnx
            Default: astore
        model_type : string, optional
            Specifies how to deploy the embedding model. "branch" means only one branch model is deployed to extract
            features while "full" means the full model is deployed to extract features for all branches and
            compute the distance metric values for all input data pairs.
            Valid values: branch and full
            Default: branch

        Notes
        -----
        Currently, this function supports sashdat, astore, and onnx formats.

        More information about ONNX can be found at: https://onnx.ai/

        DLPy supports ONNX version >= 1.3.0, and Opset version 8.

        For ONNX format, currently supported layers are convo, pool,
        fc, batchnorm, residual, concat, reshape, and detection.

        If dropout is specified in the model, train the model using
        inverted dropout, which can be specified in :class:`Optimizer`.
        This will ensure the results are correct when running the
        model during test phase.

        Returns
        --------
        :class:`Model` for a branch model when model_type is 'branch'

        """

        if model_type.lower() not in ['branch', 'full']:
            raise DLPyError('Only branch and full are valid.')

        if model_type.lower() == 'full':
            temp_embed_layers = []
            for i_branch in range(self.number_of_branches):
                temp_embed_layers.append(self.embedding_layer_name_prefix + str(i_branch))
            self.deploy(path=path, output_format=output_format, layers=temp_embed_layers)
        else:
            # create a fake task layer
            fake_output_layer = OutputLayer(n=1, name='Output1')(self.branch_output_tensor)
            # build the branch model from the tensor
            branch_model = Model(self.conn, inputs=self.branch_input_tensor, outputs=fake_output_layer,
                                 model_table=self.model_name + '_branch')
            branch_model.compile()

            # attach weights
            weight_tbl = WeightsTable(self.conn, self.model_weights.name, self.model_name)
            branch_model.set_weights(weight_tbl)

            # inherit the weight attr from the full model
            self.conn.retrieve('table.attribute', _messagelevel='error',
                               name=self.model_weights.name, task='CONVERT',
                               attrtable=self.model_weights.name + '_attr')

            self.conn.retrieve('table.attribute', _messagelevel='error',
                               name=branch_model.model_weights.name, task='ADD',
                               attrtable=self.model_weights.name + '_attr')

            self.conn.retrieve('table.dropTable', _messagelevel='error',
                               table=self.model_weights.name + '_attr')

            # add model attrs
            data_specs = [DataSpec(type_='IMAGE', layer=self.input_layer_name_prefix + '0', data=['_image_']),
                          DataSpec(type_='NUMNOM', layer='Output1', data=['_fake_output_'],
                                   nominals=['_fake_output_'])]
            create_extended_attributes(branch_model.conn, branch_model.model_name, branch_model.layers, data_specs)

            # deploy it
            temp_embed_layer = self.embedding_layer_name_prefix + '0'
            branch_model.deploy(path=path, output_format=output_format,
                                layers=temp_embed_layer)

            return branch_model
