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

''' A Network is way to compose layers: the topological form of a Model '''

import os

from dlpy.layers import Layer
from dlpy.utils import DLPyError, input_table_check, random_name, check_caslib, get_server_path_sep, \
    underscore_to_camelcase, caslibify_context, isnotebook, file_exist_on_server
from .layers import InputLayer, Conv2d, Pooling, BN, Res, Concat, Dense, OutputLayer, Keypoints, Detection, Scale, \
    Reshape, GroupConv2d, ChannelShuffle, RegionProposal, ROIPooling, FastRCNN, Conv2DTranspose, Recurrent, \
    LayerNormalization, MultiHeadAttention, Survival, EmbeddingLoss, Segmentation, FCMPLayer, Clustering, Split
import dlpy.model
import collections
import pandas as pd
import swat as sw
from copy import deepcopy
from swat.cas.table import CASTable
from . import __dev__

UNSUPPORTED_EXTRACT_LAYER = {20: "FULLCONNECTCAP", 21: "NORMCAP", 30: "ROIALIGN", 31: "NAS_LAYER",
                             32: "MASKRCNN"
                             }


class Network(Layer):
    '''
    Network

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    inputs: iter-of-Layers, optional
        Specifies some input layer(s) to instantiate a Network
    outputs: iter-of-Layers, optional
        Specifies some output layer(s) to instantiate a Network
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default: None
    model_weights : CASTable or string or dict, optional
        Specifies the CASTable containing weights of the deep learning model.
        If not specified, random initial will be used.
        Default: None
    Returns
    -------
    :class:`Model`

    '''

    type = 'model'
    type_label = 'Model'
    type_desc = 'Model'
    can_be_last_layer = True
    number_of_instances = 0
    src_layers = []
    name = 'model' + str(number_of_instances)

    def __init__(self, conn, inputs=None, outputs=None, model_table=None, model_weights=None):
        if (inputs is None or outputs is None) and (inputs is not None or outputs is not None):
            raise ValueError('If one of inputs and outputs option is enabled, both should be specified')
        self._init_model(conn, model_table, model_weights)
        # works for Sequential() as well as objects that inherit the Model class
        if isinstance(self, dlpy.model.Model):
            # 1). Model(s, model_table, model_weights)
            # 2). Model(s, inp, outputs, model_table, model_weights)
            if all(i is not None for i in [inputs, outputs]):
                self._map_graph_network(inputs, outputs)

    def _init_model(self, conn, model_table=None, model_weights=None):
        conn.loadactionset(actionSet='deeplearn', _messagelevel='error')

        self.conn = conn

        if model_table is None:
            model_table = dict(name=random_name('Model', 6))

        model_table_opts = input_table_check(model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name=random_name('Model', 6)))

        self.model_name = model_table_opts['name']
        self.model_table = model_table_opts

        if model_weights is None:
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))
        else:
            # TODO put tableexits in set_weights
            if self.conn.tableexists(model_weights).exists == 1:
                self.set_weights(model_weights)
            else:
                self.model_weights = self.conn.CASTable(**input_table_check(model_weights))

        self.layers = []
        self.model_type = 'CNN'
        self.best_weights = None
        self.target = None
        self.num_params = None
        self.count_instances()
        self.model_counter = 0

    def _map_graph_network(self, inputs, outputs):
        '''
        Propagate all of layers

        inputs : iter-of-Tensors
        outputs : iter-of-Tensors

        '''

        def build_map(start):
            if start.name is None:
                start.count_instances()
                start.name = str(start.__class__.__name__) + '_' + str(type(start).number_of_instances)
            # if the node is visited, continue
            if start in self.layers:
                return
            # if the node is an input layer, add it and return
            if start in self.input_layers and start.type != 'model':
                self.layers.append(start)
                return
            for src_layer in start.src_layers:
                build_map(src_layer)
                # if all of src_layer of layer is input layer type, add it in layers list
                src_layer.depth = 0 if src_layer.type == 'input' \
                    else max([i.depth for i in src_layer.src_layers]) + 1

                if all(i in self.layers for i in start.src_layers) and start not in self.layers:
                    self.layers.append(start)
            return

        if not isinstance(inputs, collections.Iterable):
            inputs = [inputs]
        if any(x.__class__.__name__ != 'Tensor' for x in inputs):
            raise DLPyError('All inputs should be tensors.')
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]

        self.inputs = inputs
        self.outputs = outputs
        self.output_layers = [output._op for output in outputs]
        self.input_layers = [input._op for input in inputs]

        for layer in self.output_layers:
            build_map(layer)
            layer.depth = max([i.depth for i in layer.src_layers]) + 1

        return

    def compile(self):
        ''' parse the network nodes and process CAS Action '''
        for l in self.layers:
            if isinstance(l, Recurrent):
                self.model_type = 'RNN'
        rt = self._retrieve_('deeplearn.buildmodel',
                             model=dict(name=self.model_name, replace=True), type=self.model_type)

        if rt.severity > 1:
            raise DLPyError('cannot build model, there seems to be a problem.')
        self.num_params = 0

        # before addLayer, self.layers are reordered based on layer_id
        # the new id is created based on model.summary which contains layer_id.
        # since it is not compiled, it might contains None or index from it original model summary.
        orig_ids = self.summary['Layer Id']
        offset_id = 0
        new_ids = []  # store new ids
        pool = []  # pool to store index has been added
        for idx, l_id in enumerate(orig_ids):
            # encounter None id or duplicated id, start a new Model
            if l_id is None or l_id in pool:
                offset_id = idx
                pool = []
            else:
                pool.append(l_id)

            # calculate new index for the layer
            if l_id is None:
                l_id = offset_id
            else:
                l_id += offset_id

            new_ids.append(l_id)

        # reassign layer_id
        for i, l in enumerate(self.layers):
            l.layer_id = new_ids[i]
        self.layers.sort()  # sort based on layer_id

        for layer in self.layers:
            option = layer.to_model_params()
            rt = self._retrieve_('deeplearn.addlayer', model=self.model_name, **option)
            if rt.severity > 1:
                raise DLPyError('there seems to be an error while adding the ' + str(layer.name) + '.')
            if layer.num_weights is None:
                num_weights = 0
            else:
                num_weights = layer.num_weights

            if layer.num_bias is None:
                num_bias = 0
            else:
                num_bias = layer.num_bias

            self.num_params += num_weights + num_bias
        print('NOTE: Model compiled successfully.')

    def to_functional_model(self, stop_layers=None):
        '''
        Convert a Sequential into a functional model and return the functional model.

        stop_layers : iter-of-Layer or Layer
            stop_layers refers to the layers that stop traverse the graph.
            All of layers followed by the stop_layers are removed from the functional model.
            The argument is useful when you want to get a subset of network.
            For example:
                Given a ResNet50 model, only generate the feature extraction network of ResNet50.
                feature_extractor = resnet50_model.to_functional_model(stop_layers=resnet50_model.layers[-1])

        Returns
        -------
        :class:`Model`

        '''

        def _if_traverse_to_an_input(l):
            # depth first traverse
            for l_src in l.src_layers:
                # if the layer is input layer return true
                if l_src.type == 'input':
                    return True
                # encounter stop_layers, continue
                if l_src in stop_layers:
                    continue
                # as long as encounter one input layer return true
                if _if_traverse_to_an_input(l_src):
                    return True

        copied_model = deepcopy(self)  # deepcopy the sequential model and don't touch the original one
        stop_layers = stop_layers or []
        input_tensors = []
        output_tensors = []

        if not isinstance(stop_layers, collections.Iterable):
            stop_layers = [stop_layers]
        index_l = [self.layers.index(x) for x in stop_layers]
        stop_layers = [copied_model.layers[i] for i in index_l]
        # remove tensor attribute if exist except for InputLayer
        for l in copied_model.layers:
            if l.__class__.__name__ != 'InputLayer':
                if hasattr(l, 'tensor'):
                    delattr(l, 'tensor')

        for idx, layer in enumerate(copied_model.layers):
            layer_type = layer.__class__.__name__
            if layer_type == 'InputLayer':
                input_tensors.append(layer.tensor)
                continue
            # find layer's outbound layer
            for outbound_layer in copied_model.layers[idx:]:
                if outbound_layer.__class__.__name__ == 'InputLayer':
                    continue
                # if all source layers of outbound_layer are visited(all in self.layers[:idx])
                if all(src_layer in copied_model.layers[:idx] for src_layer in outbound_layer.src_layers):
                    # skip if stop_layers are visited
                    if outbound_layer in stop_layers:
                        # add src_layers's into output tensors
                        for src_layer in outbound_layer.src_layers:
                            # if src_layer doesn't have tensor attribute which means it dependency cannot be loaded.
                            # so it is not a valid output_tensors.
                            if hasattr(src_layer, 'tensor'):
                                output_tensors.append(src_layer.tensor)
                        continue
                    # initialize tensor of the outbound_layer
                    # if any of tensor doesn't exit, stop calling the layer
                    try:
                        outbound_layer([l.tensor for l in outbound_layer.src_layers])
                    except AttributeError:
                        continue
                    if outbound_layer.can_be_last_layer:
                        # see if output tensor can traverse to an input layer
                        # if removing connections related to stop layers
                        if _if_traverse_to_an_input(outbound_layer):
                            output_tensors.append(outbound_layer.tensor)
                            break

        return dlpy.model.Model(self.conn, input_tensors, output_tensors)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'conn':
                continue
            setattr(result, k, deepcopy(v, memo))
        return result

    def _retrieve_(self, _name_, message_level='error', **kwargs):
        ''' Call a CAS action '''
        return self.conn.retrieve(_name_, _messagelevel=message_level, **kwargs)

    @classmethod
    def count_instances(cls):
        cls.number_of_instances += 1

    @classmethod
    def from_table(cls, input_model_table, display_note=True, output_model_table=None):
        '''
        Create a Model object from CAS table that defines a deep learning model

        Parameters
        ----------
        input_model_table : CASTable
            Specifies the CAS table that defines the deep learning model.
        display_note : bool, optional
            Specifies whether to print the note when generating the model table.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        model = cls(conn=input_model_table.get_connection(), model_table=output_model_table)
        model_name = model._retrieve_('table.fetch',
                                      table=dict(where='_DLKey1_= "modeltype"',
                                                 **input_model_table.to_table_params()))
        model_name = model_name.Fetch['_DLKey0_'][0]
        if display_note:
            print(('NOTE: Model table is attached successfully!\n'
                   'NOTE: Model is named to "{}" according to the '
                   'model name in the table.').format(model_name))
        model.model_name = model_name
        model.model_table.update(**input_model_table.to_table_params())
        model.model_weights = model.conn.CASTable('{}_weights'.format(model_name))

        model_table = input_model_table.to_frame()
        for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
            layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
            layer_type = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                   'layertype'].tolist()[0]
            if layer_type == 1:
                model.layers.append(extract_input_layer(layer_table=layer_table))
            elif layer_type == 2:
                model.layers.append(extract_conv_layer(layer_table=layer_table))
            elif layer_type == 3:
                model.layers.append(extract_pooling_layer(layer_table=layer_table))
            elif layer_type == 4:
                model.layers.append(extract_fc_layer(layer_table=layer_table))
            elif layer_type == 5:
                model.layers.append(extract_output_layer(layer_table=layer_table))
            elif layer_type == 6:
                model.layers.append(extract_recurrent_layer(layer_table=layer_table))
            elif layer_type == 8:
                model.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layer_type == 9:
                model.layers.append(extract_residual_layer(layer_table=layer_table))
            elif layer_type == 10:
                model.layers.append(extract_concatenate_layer(layer_table=layer_table))
            elif layer_type == 11:
                model.layers.append(extract_detection_layer(layer_table=layer_table))
            elif layer_type == 12:
                model.layers.append(extract_scale_layer(layer_table=layer_table))
            elif layer_type == 13:
                model.layers.append(extract_keypoints_layer(layer_table=layer_table))
            elif layer_type == 14:
                model.layers.append(extract_reshape_layer(layer_table = layer_table))
            elif layer_type == 15:
                model.layers.append(extract_fcmp_layer(layer_table = layer_table))
            elif layer_type == 16:
                model.layers.append(extract_conv2dtranspose_layer(layer_table=layer_table))
            elif layer_type == 17:
                model.layers.append(extract_groupconv_layer(layer_table=layer_table))
            elif layer_type == 18:
                model.layers.append(extract_channelshuffle_layer(layer_table = layer_table))
            elif layer_type == 19:
                model.layers.append(extract_segmentation_layer(layer_table = layer_table))
            elif layer_type == 22:
                model.layers.append(extract_embeddingloss_layer(layer_table = layer_table))
            elif layer_type == 23:
                model.layers.append(extract_rpn_layer(layer_table=layer_table))
            elif layer_type == 24:
                model.layers.append(extract_roipooling_layer(layer_table=layer_table))
            elif layer_type == 25:
                model.layers.append(extract_fastrcnn_layer(layer_table = layer_table))
            elif layer_type == 26:
                model.layers.append(extract_cluster_layer(layer_table = layer_table))
            elif layer_type == 27:
                model.layers.append(extract_survival_layer(layer_table = layer_table))
            elif layer_type == 28:
                model.layers.append(extract_layernorm_layer(layer_table=layer_table))
            elif layer_type == 29:
                model.layers.append(extract_mhattention_layer(layer_table = layer_table))
            elif layer_type == 33:
                model.layers.append(extract_split_layer(layer_table = layer_table))
            else:
                raise DLPyError("Extracting Layer type, {}, is not"
                                " supported yet.".format(UNSUPPORTED_EXTRACT_LAYER[layer_type]))

        conn_mat = model_table[['_DLNumVal_', '_DLLayerID_']][
            model_table['_DLKey1_'].str.contains('srclayers')].sort_values('_DLLayerID_')
        layer_id_list = conn_mat['_DLLayerID_'].tolist()
        src_layer_id_list = conn_mat['_DLNumVal_'].tolist()

        for row_id in range(conn_mat.shape[0]):
            layer_id = int(layer_id_list[row_id])
            src_layer_id = int(src_layer_id_list[row_id])
            if model.layers[layer_id].src_layers is None:
                model.layers[layer_id].src_layers = [model.layers[src_layer_id]]
            else:
                model.layers[layer_id].src_layers.append(model.layers[src_layer_id])

        return model

    @classmethod
    def from_sashdat(cls, conn, path, output_model_table=None):
        '''
        Generate a model object using the model information in the sashdat file

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        path : string
            The path of the sashdat file, the path has to be accessible
            from the current CAS session.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        model = cls(conn, model_table=output_model_table)
        model.load(path=path)
        return model

    @classmethod
    def from_caffe_model(cls, conn, input_network_file, output_model_table=None,
                         model_weights_file=None, **kwargs):
        '''
        Generate a model object from a Caffe model proto file (e.g. *.prototxt), and
        convert the weights (e.g. *.caffemodel) to a SAS capable file (e.g. *.caffemodel.h5).

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        input_network_file : string
            Fully qualified file name of network definition file (*.prototxt).
        model_weights_file : string, optional
            Fully qualified file name of model weights file (*.caffemodel)
            Default: None
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None

        Returns
        -------
        :class:`Model`

        '''
        from .model_conversion.sas_caffe_parse import caffe_to_sas

        if output_model_table is None:
            output_model_table = dict(name=random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name=random_name('caffe_model', 6)))

        model_name = model_table_opts['name']

        output_code = caffe_to_sas(input_network_file, model_name, network_param=model_weights_file, **kwargs)
        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table=input_model_table)
        return model

    @classmethod
    def from_keras_model(cls, conn, keras_model, output_model_table=None,
                         offsets=None, std=None, scale=1.0,
                         max_num_frames=-1, include_weights=False,
                         input_weights_file=None, verbose=False):
        '''
        Generate a model object from a Keras model object

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        keras_model : keras_model object
            Specifies the keras model to be converted.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        offsets : list, optional
            Specifies the values to be subtracted from the pixel values
            of the input data, used if the data is an image.
        std : list or None
            The pixel values of the input data are divided by these
            values, used if the data is an image.
        scale : float, optional
            Specifies the scaling factor to apply to each image.
        max_num_frames : int, optional
            Maximum number of frames for sequence processing.
        include_weights : bool, optional
            Specifies whether to load the weights of the keras model.
            Default: True
        input_weights_file : string, optional
            A fully specified client side path to the HDF5 file that stores
            the keras model weights. Only effective when include_weights=True.
            If None is given, the current weights in the keras model will be used.
            Default: None
        verbose : boolean optional
            Specifies whether to print warning messages and debugging information
            Default: False

        Returns
        -------
        :class:`Model`
        boolean : use GPU

        '''

        from .model_conversion.sas_keras_parse import keras_to_sas
        if output_model_table is None:
            output_model_table = dict(name=random_name('keras_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name=random_name('keras_model', 6)))

        model_name = model_table_opts['name']

        # determine what features are supported by current Viya server/deep learning action set
        from .model_conversion.model_conversion_utils import check_rnn_import, check_normstd
        conn.loadactionset('deeplearn', _messagelevel='error')
        rnn_support = check_rnn_import(conn)
        normstd_support = check_normstd(conn)
        if (std is not None) and (not normstd_support):
            print('WARNING: Your Viya installation does not support the std parameter - ignoring')
            std = None

        output_code = keras_to_sas(model=keras_model, rnn_support=rnn_support,
                                   model_name=model_name, offsets=offsets, std=std,
                                   scale=scale, max_num_frames=max_num_frames, verbose=verbose)

        if verbose:
            print(output_code)

        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table=input_model_table)

        use_gpu = False
        if include_weights:
            from .model_conversion.write_keras_model_parm import write_keras_hdf5, write_keras_hdf5_from_file
            temp_HDF5 = os.path.join(os.getcwd(), '{}_weights.kerasmodel.h5'.format(model_name))
            if input_weights_file is None:
                use_gpu = write_keras_hdf5(keras_model, rnn_support, temp_HDF5)
            else:
                use_gpu = write_keras_hdf5_from_file(keras_model, rnn_support, input_weights_file, temp_HDF5)
            print('NOTE: the model weights has been stored in the following file:\n'
                  '{}'.format(temp_HDF5))

        return model, use_gpu

    @classmethod
    def from_onnx_model(cls, conn, onnx_model, output_model_table=None,
                        offsets=None, scale=None, std=None, norm_stds=None, output_layer=None):
        '''
        Generate a Model object from ONNX model.

        Parameters
        ----------
        conn : CAS
            Specifies the CAS connection object.
        onnx_model : ModelProto
            Specifies the ONNX model.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        offsets : int-list, optional
            Specifies the values to be subtracted from the pixel values
            of the input data, used if the data is an image.
        scale : float, optional
            Specifies the scaling factor to apply to each image.
        std : string, optional
            Specifies how to standardize the variables in the input layer.
            Valid Values: MIDRANGE, NONE, STD
        norm_stds : float-list, optional
            Specifies a standard deviation for each channel in the input data.
            The final input data is normalized with specified means and standard deviations.
        output_layer : Layer object, optional
            Specifies the output layer of the model. If no output
            layer is specified, the last layer is automatically set
            as :class:`OutputLayer` with SOFTMAX activation.

        Returns
        -------
        :class:`Model`

        '''

        from .model_conversion.sas_onnx_parse import onnx_to_sas
        if output_model_table is None:
            output_model_table = dict(name=random_name('onnx_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name=random_name('onnx_model', 6)))

        model_name = model_table_opts['name']

        _layers = onnx_to_sas(onnx_model, model_name, output_layer)
        if offsets is not None:
            _layers[0].config.update(offsets=offsets)
        if scale is not None:
            _layers[0].config.update(scale=scale)
        if std is not None:
            _layers[0].config.update(std=std)
        if norm_stds is not None:
            _layers[0].config.update(norm_stds=norm_stds)
        if len(_layers) == 0:
            raise DLPyError('Unable to import ONNX model.')

        conn.loadactionset('deeplearn', _messagelevel='error')
        rt = conn.retrieve('deeplearn.buildmodel',
                           _messagelevel='error',
                           model=dict(name=model_name, replace=True),
                           type='CNN')
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('Cannot build model, there seems to be a problem.')

        for layer in _layers:
            option = layer.to_model_params()
            rt = conn.retrieve('deeplearn.addlayer', _messagelevel='error',
                               model=model_name, **option)
            if rt.severity > 1:
                for m in rt.messages:
                    print(m)
                raise DLPyError('There seems to be an error while adding the '
                                + layer.name + '.')

        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table=input_model_table)
        print('NOTE: Successfully imported ONNX model.')
        return model

    @property
    def summary(self):
        if self.model_type == 'CNN':
            return pd.concat([x.summary for x in self.layers], ignore_index=True)
        else:
            return pd.concat([x.rnn_summary for x in self.layers], ignore_index=True)

    def __load_layer_ids(self):
        import math
        try:
            # only check each layer once
            model_table_rows = self.conn.table.fetch(table=dict(self.model_table, where='_DLKey1_ eq "layertype"'),
                                                     maxrows=1000000, to=1000000).Fetch
        except:
            model_table_rows = None
        if model_table_rows is not None:
            layer_ids = {}
            for index, row in model_table_rows.iterrows():
                if not math.isnan(row['_DLLayerID_']):
                    layer_ids[row['_DLKey0_']] = int(row['_DLLayerID_'])

            for l in self.layers:
                l.layer_id = layer_ids[l.name.lower()]

    def print_summary(self):
        '''

        Display a table that summarizes the model architecture

        Returns
        -------
        :pandas data frame

        '''

        try:
            if len(self.layers) > 0 and self.layers[0].layer_id is None:
                self.__load_layer_ids()

            from IPython.display import display

            if self.model_type == 'CNN':
                # generate layers' summary before getting total values.
                layers_summary = self.summary
                if self.num_params is None:
                    self.num_params = 0
                    for l in self.layers:
                        if l.num_weights is not None:
                            self.num_params += l.num_weights
                        if l.num_bias is not None:
                            self.num_params += l.num_bias
                num_params_str = format(self.num_params, ",d")  # value with comma

                total_FLOPS = 0
                for l in self.layers:
                    if l.FLOPS:
                        total_FLOPS += l.FLOPS
                total_FLOPS = format(total_FLOPS, ",d")  # value with comma
                MB = 2 ** 20
                # total summary rows
                total = pd.DataFrame([['', '', '', '', '', '', '', 'Total number of parameters', 'Total FLOPS'],
                                      ['Summary', '', '', '', '', '', '', num_params_str, total_FLOPS]],
                                     columns=['Layer Id', 'Layer', 'Type', 'Kernel Size', 'Stride',
                                              'Activation', 'Output Size', 'Number of Parameters',
                                              'FLOPS(forward pass)'])
                pd_layers = pd.concat([layers_summary, total], ignore_index=True)
                if not isnotebook():
                    display(pd_layers)
                return pd_layers
            else:
                if not isnotebook():
                    display(self.summary)
                return self.summary
        except ImportError:
            if not isnotebook():
                print(self.summary)
            return self.summary

    def _repr_html_(self):
        return self.summary._repr_html_()

    def plot_network(self):
        '''
        Display a graph that summarizes the model architecture.

        Returns
        -------
        :class:`graphviz.dot.Digraph`

        '''
        return model_to_graph(self)

    def _repr_svg_(self):
        return self.plot_network()._repr_svg_()

    def set_weights(self, weight_tbl):
        '''
        Assign weights to the Model object

        Parameters
        ----------
        weight_tbl : CASTable or string or dict or WeightsTable
            Specifies the weights CAS table for the model

        '''
        weight_name = self.model_name + '_weights'
        # if weights_tbl is WeightsTable, we will remap layer id if necessary
        # remapping is needed when the model's layer id mapping is different from original model's.
        if type(weight_tbl) == WeightsTable:
            model_mapper = self.create_layer_id_name_mapping()
            # check if need to remap
            if weight_tbl.weights_mapping != model_mapper:
                weight_tbl.remap_layer_ids(model_mapper, weight_name)
            weight_tbl = dict(name=weight_name)
            if weight_tbl['name'].lower() != weight_name.lower():
                self.conn.altertable(name='weight_name.lower()', rename=weight_name)
        else:
            weight_tbl = input_table_check(weight_tbl)
            if weight_tbl['name'].lower() != weight_name.lower():
                self._retrieve_('table.partition',
                                casout=dict(replace=True, name=weight_name),
                                table=weight_tbl)

        self.model_weights = self.conn.CASTable(name=weight_name)
        if self.conn.tableexists(weight_name).exists:
            print('NOTE: Model weights attached successfully!')
        else:
            raise DLPyError('Model weights attached unsuccessfully!')

    def load(self, path, display_note=True):
        '''
        Load the deep learning model architecture from existing table

        Parameters
        ----------
        path : string
            Specifies the absolute server-side path of the table file.
        display_note : bool
            Specifies whether to print the note when generating the model table.

        '''

        with caslibify_context(self.conn, path, task='load') as (cas_lib_name, file_name):
            self._retrieve_('table.loadtable',
                            caslib=cas_lib_name,
                            path=file_name,
                            casout=dict(replace=True, **self.model_table))

            model_name = self._retrieve_('table.fetch',
                                         table=dict(where='_DLKey1_= "modeltype"',
                                                    **self.model_table)).Fetch['_DLKey0_'][0]

            if model_name.lower() != self.model_name.lower():
                self._retrieve_('table.partition',
                                casout=dict(replace=True, name=model_name),
                                table=self.model_name)

                self._retrieve_('table.droptable', **self.model_table)
                if display_note:
                    print(('NOTE: Model table is loaded successfully!\n'
                           'NOTE: Model is renamed to "{}" according to the '
                           'model name in the table.').format(model_name))
                self.model_name = model_name
                self.model_table['name'] = model_name
                self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))

            model_table = self.conn.CASTable(self.model_name).to_frame()
            for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
                layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
                layer_type = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                       'layertype'].tolist()[0]
                if layer_type == 1:
                    self.layers.append(extract_input_layer(layer_table=layer_table))
                elif layer_type == 2:
                    self.layers.append(extract_conv_layer(layer_table=layer_table))
                elif layer_type == 3:
                    self.layers.append(extract_pooling_layer(layer_table=layer_table))
                elif layer_type == 4:
                    self.layers.append(extract_fc_layer(layer_table=layer_table))
                elif layer_type == 5:
                    self.layers.append(extract_output_layer(layer_table=layer_table))
                elif layer_type == 6:
                    self.layers.append(extract_recurrent_layer(layer_table=layer_table))
                elif layer_type == 8:
                    self.layers.append(extract_batchnorm_layer(layer_table=layer_table))
                elif layer_type == 9:
                    self.layers.append(extract_residual_layer(layer_table=layer_table))
                elif layer_type == 10:
                    self.layers.append(extract_concatenate_layer(layer_table=layer_table))
                elif layer_type == 11:
                    self.layers.append(extract_detection_layer(layer_table=layer_table))
                elif layer_type == 12:
                    self.layers.append(extract_scale_layer(layer_table=layer_table))
                elif layer_type == 13:
                    self.layers.append(extract_keypoints_layer(layer_table=layer_table))
                elif layer_type == 14:
                    self.layers.append(extract_reshape_layer(layer_table = layer_table))
                elif layer_type == 15:
                    self.layers.append(extract_fcmp_layer(layer_table = layer_table))
                elif layer_type == 16:
                    self.layers.append(extract_conv2dtranspose_layer(layer_table=layer_table))
                elif layer_type == 17:
                    self.layers.append(extract_groupconv_layer(layer_table=layer_table))
                elif layer_type == 18:
                    self.layers.append(extract_channelshuffle_layer(layer_table = layer_table))
                elif layer_type == 19:
                    self.layers.append(extract_segmentation_layer(layer_table = layer_table))
                elif layer_type == 22:
                    self.layers.append(extract_embeddingloss_layer(layer_table = layer_table))
                elif layer_type == 23:
                    self.layers.append(extract_rpn_layer(layer_table=layer_table))
                elif layer_type == 24:
                    self.layers.append(extract_roipooling_layer(layer_table=layer_table))
                elif layer_type == 25:
                    self.layers.append(extract_fastrcnn_layer(layer_table = layer_table))
                elif layer_type == 26:
                    self.layers.append(extract_cluster_layer(layer_table = layer_table))
                elif layer_type == 27:
                    self.layers.append(extract_survival_layer(layer_table = layer_table))
                elif layer_type == 28:
                    self.layers.append(extract_layernorm_layer(layer_table=layer_table))
                elif layer_type == 29:
                    self.layers.append(extract_mhattention_layer(layer_table = layer_table))
                elif layer_type == 33:
                    self.layers.append(extract_split_layer(layer_table = layer_table))
                else:
                    raise DLPyError("Extracting Layer type, {}, is not"
                                    " supported yet.".format(UNSUPPORTED_EXTRACT_LAYER[layer_type]))

            conn_mat = model_table[['_DLNumVal_', '_DLLayerID_']][
                model_table['_DLKey1_'].str.contains('srclayers')].sort_values('_DLLayerID_')
            layer_id_list = conn_mat['_DLLayerID_'].tolist()
            src_layer_id_list = conn_mat['_DLNumVal_'].tolist()

            for row_id in range(conn_mat.shape[0]):
                layer_id = int(layer_id_list[row_id])
                src_layer_id = int(src_layer_id_list[row_id])
                if self.layers[layer_id].src_layers is None:
                    self.layers[layer_id].src_layers = [self.layers[src_layer_id]]
                else:
                    self.layers[layer_id].src_layers.append(self.layers[src_layer_id])

            # Check if weight table is in the same path
            _file_name_, _extension_ = os.path.splitext(file_name)

            _file_name_list_ = list(self._retrieve_('table.fileinfo',
                                                    caslib=cas_lib_name,
                                                    includeDirectories=False).FileInfo.Name)

            if (_file_name_ + '_weights' + _extension_) in _file_name_list_:
                print('NOTE: ' + _file_name_ + '_weights' + _extension_ +
                      ' is used as model weigths.')

                self._retrieve_('table.loadtable',
                                caslib=cas_lib_name,
                                path=_file_name_ + '_weights' + _extension_,
                                casout=dict(replace=True, name=self.model_name + '_weights'))
                self.set_weights(self.model_name + '_weights')

                if (_file_name_ + '_weights_attr' + _extension_) in _file_name_list_:
                    print('NOTE: ' + _file_name_ + '_weights_attr' + _extension_ +
                          ' is used as weigths attribute.')
                    self._retrieve_('table.loadtable',
                                    caslib=cas_lib_name,
                                    path=_file_name_ + '_weights_attr' + _extension_,
                                    casout=dict(replace=True,
                                                name=self.model_name + '_weights_attr'))
                    self.set_weights_attr(self.model_name + '_weights_attr')

    def load_weights(self, path, labels=False, data_spec=None, label_file_name=None, label_length=None,
                     use_gpu=False, embedding_dim=None):
        '''
        Load the weights from a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.
        labels : bool
            Specifies whether to apply user-defined classification labels
            Default: False
        data_spec : list of :class:`DataSpec`, optional
            data specification for input and output layer(s)
            Default: None
        label_file_name : string, optional
            Fully qualified path to CSV file containing user-defined
            classification labels.  If not specified, ImageNet labels assumed.
            Default: None
        label_length : int, optional
            Length of the classification labels (in characters).
            Default: None
        use_gpu: boolean, optional
            GPU processing of model required (or not)
            Default: False
        embedding_dim : int, optional
            Specifies text embedding dimension.  You must specify the data_spec parameter
            or this parameter is ignored.
            Default: None

        Notes
        -----
        Currently support HDF5 and sashdat files.

        '''
        if not file_exist_on_server(self.conn, path):
            raise DLPyError('The file, {}, doesn\'t exist on the server-side.'.format(path))

        server_sep = get_server_path_sep(self.conn)

        if server_sep in path:
            dir_name, file_name = path.rsplit(server_sep, 1)
        else:
            file_name = path

        if file_name.lower().endswith('.sashdat'):
            self.load_weights_from_table(path)
        elif file_name.lower().endswith('caffemodel.h5'):
            self.load_weights_from_caffe(path, labels=labels, data_spec=data_spec, label_file_name=label_file_name,
                                         label_length=label_length)
        elif file_name.lower().endswith('kerasmodel.h5'):
            self.load_weights_from_keras(path, labels=labels, data_spec=data_spec, label_file_name=label_file_name,
                                         label_length=label_length, use_gpu=use_gpu, embedding_dim=embedding_dim)
        elif file_name.lower().endswith('onnxmodel.h5'):
            self.load_weights_from_keras(path, labels=labels, data_spec=data_spec, label_file_name=label_file_name,
                                         label_length=label_length, use_gpu=use_gpu, embedding_dim=embedding_dim)
        else:
            raise DLPyError('Weights file must be one of the follow types:\n'
                            'sashdat, caffemodel.h5 or kerasmodel.h5.\n'
                            'Weights load failed.')

    def load_weights_from_caffe(self, path, labels=False, data_spec=None, label_file_name=None, label_length=None):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        labels : bool
            Specifies whether to use ImageNet classification labels
        data_spec : list of :class:`DataSpec`, optional
            data specification for input and output layer(s)
        label_file_name : string, optional
            Fully qualified path to CSV file containing user-defined
            classification labels.  If not specified, ImageNet labels assumed.
        label_length : int, optional
            Length of the classification labels (in characters).

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='CAFFE', data_spec=data_spec,
                                                    label_file_name=label_file_name, label_length=label_length)
        else:
            self.load_weights_from_file(path=path, format_type='CAFFE', data_spec=data_spec)

    def load_weights_from_keras(self, path, labels=False, data_spec=None, label_file_name=None, label_length=None,
                                use_gpu=False, embedding_dim=None):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        labels : bool
            Specifies whether to use ImageNet classification labels
            Default: False
        data_spec : list of :class:`DataSpec`, optional
            data specification for input and output layer(s)
            Default: None
        label_file_name : string, optional
            Fully qualified path to CSV file containing user-defined
            classification labels.  If not specified, ImageNet labels assumed.
            Default: None
        label_length : int, optional
            Length of the classification labels (in characters).
            Default: None
        use_gpu : boolean, optional
            Require GPU for processing model
            Default: False
        embedding_dim : int, optional
            Specifies text embedding dimension.  You must specify the data_spec parameter
            or this parameter is ignored.
            Default: None
            

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='KERAS', data_spec=data_spec,
                                                    label_file_name=label_file_name, label_length=label_length,
                                                    use_gpu=use_gpu, embedding_dim=embedding_dim)
        else:
            self.load_weights_from_file(path=path, format_type='KERAS', data_spec=data_spec, use_gpu=use_gpu,
                                        embedding_dim=embedding_dim)

    def load_weights_from_file(self, path, format_type='KERAS', data_spec=None, use_gpu=False, embedding_dim=None):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file
        data_spec : list of :class:`DataSpec`, optional
            Data specification for input and output layer(s)
            Default: None
        use_gpu : boolean, optional
            Require GPU for processing model
            Default: False
        embedding_dim : int, optional
            Specifies text embedding dimension.  You must specify the data_spec parameter
            or this parameter is ignored.
            Default: None

        '''
        from dlpy.model_conversion.model_conversion_utils import query_action_parm

        with caslibify_context(self.conn, path, task='load') as (cas_lib_name, file_name):
            has_gpu_model, act_parms = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn', 'gpuModel')
            if (not has_gpu_model) and use_gpu:
                raise DLPyError('A GPU model was specified, but your Viya installation does not support'
                                'importing GPU models.')

            if data_spec:

                has_data_spec = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn', 'gpuModel')

                has_embedding_dim = False
                if embedding_dim is not None:
                    has_embedding_dim = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn',
                                                          'textEmbeddingDim')
                    if not has_embedding_dim:
                        raise DLPyError('A text embedding dimension was specified, but your Viya installation does not'
                                        'support this parameter.')

                if has_data_spec:
                    # run action with dataSpec option
                    if has_gpu_model and (not has_embedding_dim):
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 gpuModel=use_gpu,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name)
                    elif (not has_gpu_model) and (not has_embedding_dim):
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name)
                    elif has_gpu_model and has_embedding_dim:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 gpuModel=use_gpu,
                                                 textEmbeddingDim=embedding_dim,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name)
                    else:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 textEmbeddingDim=embedding_dim,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name)

                else:
                    if has_gpu_model:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                                 modelWeights=dict(replace=True,
                                                                   name=self.model_name + '_weights'),
                                                 formatType=format_type, weightFilePath=file_name,
                                                 gpuModel=use_gpu,
                                                 caslib=cas_lib_name)
                    else:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                                 modelWeights=dict(replace=True,
                                                                   name=self.model_name + '_weights'),
                                                 formatType=format_type, weightFilePath=file_name,
                                                 caslib=cas_lib_name)

                # handle error or create necessary attributes
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('Cannot import model weights, there seems to be a problem.')

                # create attributes if necessary
                if not has_data_spec:
                    from dlpy.attribute_utils import create_extended_attributes
                    create_extended_attributes(self.conn, self.model_name, self.layers, data_spec)

            else:
                print("NOTE: no dataspec(s) provided - creating image classification model.")
                if has_gpu_model:
                    self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                    modelWeights=dict(replace=True,
                                                      name=self.model_name + '_weights'),
                                    formatType=format_type, weightFilePath=file_name,
                                    gpuModel=use_gpu,
                                    caslib=cas_lib_name,
                                    )
                else:
                    self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                    modelWeights=dict(replace=True,
                                                      name=self.model_name + '_weights'),
                                    formatType=format_type, weightFilePath=file_name,
                                    caslib=cas_lib_name,
                                    )

        self.set_weights(self.model_name + '_weights')

    def load_weights_from_file_with_labels(self, path, format_type='KERAS',
                                           data_spec=None,
                                           label_file_name=None,
                                           label_length=None,
                                           use_gpu=False,
                                           embedding_dim=None):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file
        data_spec : list of :class:`DataSpec`, optional
            data specification for input and output layer(s)
        label_file_name : string, optional
            Fully qualified path to CSV file containing user-defined
            classification labels.  If not specified, ImageNet labels assumed.
        label_length : int, optional
            Length of the classification labels (in characters).
        use_gpu : boolean, optional
            Require GPU for processing model
        embedding_dim : int, optional
            Specifies text embedding dimension.  You must specify the data_spec parameter
            or this parameter is ignored.
            Default: None

        '''
        from dlpy.model_conversion.model_conversion_utils import query_action_parm

        with caslibify_context(self.conn, path, task='load') as (cas_lib_name, file_name):
            has_gpu_model, act_parms = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn', 'gpuModel')
            if (not has_gpu_model) and use_gpu:
                raise DLPyError('A GPU model was specified, but your Viya installation does not support'
                                'importing GPU models.')

            if label_file_name:
                from dlpy.utils import get_user_defined_labels_table
                label_table = get_user_defined_labels_table(self.conn, label_file_name, label_length)
            else:
                from dlpy.utils import get_imagenet_labels_table
                label_table = get_imagenet_labels_table(self.conn, label_length)

            if data_spec:

                has_data_spec = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn', 'gpuModel')

                has_embedding_dim = False
                if embedding_dim is not None:
                    has_embedding_dim = query_action_parm(self.conn, 'dlImportModelWeights', 'deepLearn',
                                                          'textEmbeddingDim')
                    if not has_embedding_dim:
                        raise DLPyError('A text embedding dimension was specified, but your Viya installation does not'
                                        'support this parameter.')

                if has_data_spec:
                    # run action with dataSpec option
                    if has_gpu_model and (not has_embedding_dim):
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 gpuModel=use_gpu,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name,
                                                 labelTable=label_table)
                    elif (not has_gpu_model) and (not has_embedding_dim):
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name,
                                                 labelTable=label_table)
                    elif has_gpu_model and has_embedding_dim:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 gpuModel=use_gpu,
                                                 textEmbeddingDim=embedding_dim,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name,
                                                 labelTable=label_table)
                    else:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                                 model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 dataSpecs=data_spec,
                                                 textEmbeddingDim=embedding_dim,
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name,
                                                 labelTable=label_table)
                else:
                    if has_gpu_model:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                                 gpuModel=use_gpu,
                                                 labelTable=label_table)
                    else:
                        with sw.option_context(print_messages=False):
                            rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                                 modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                                 formatType=format_type,
                                                 weightFilePath=file_name,
                                                 caslib=cas_lib_name,
                                                 labelTable=label_table)

                # handle error or create necessary attributes
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('Cannot import model weights, there seems to be a problem.')

                # create attributes if necessary
                if not has_data_spec:
                    from dlpy.attribute_utils import create_extended_attributes
                    create_extended_attributes(self.conn, self.model_name, self.layers, data_spec)

            else:
                print("NOTE: no dataspec(s) provided - creating image classification model.")
                if has_gpu_model:
                    self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                    modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                    formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                    gpuModel=use_gpu,
                                    labelTable=label_table,
                                    )
                else:
                    self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                    modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                    formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                    labelTable=label_table,
                                    )

        self.set_weights(self.model_name + '_weights')

    def load_weights_from_table(self, path):
        '''
        Load the weights from a file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        '''
        with caslibify_context(self.conn, path, task='load') as (cas_lib_name, file_name):
            self._retrieve_('table.loadtable',
                            caslib=cas_lib_name,
                            path=file_name,
                            casout=dict(replace=True, name=self.model_name + '_weights'))

            self.set_weights(self.model_name + '_weights')

            _file_name_, _extension_ = os.path.splitext(file_name)

            _file_name_list_ = list(
                self._retrieve_('table.fileinfo', caslib=cas_lib_name,
                                includeDirectories=False).FileInfo.Name)

            if (_file_name_ + '_attr' + _extension_) in _file_name_list_:
                print('NOTE: ' + _file_name_ + '_attr' + _extension_ +
                      ' is used as weigths attribute.')
                self._retrieve_('table.loadtable',
                                caslib=cas_lib_name,
                                path=_file_name_ + '_attr' + _extension_,
                                casout=dict(replace=True,
                                            name=self.model_name + '_weights_attr'))

                self.set_weights_attr(self.model_name + '_weights_attr')

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')

    def set_weights_attr(self, attr_tbl, clear=True):
        '''
        Attach the weights attribute to the model weights

        Parameters
        ----------
        attr_tbl : CASTable or string or dict
            Specifies the CAS table that contains the weights attribute table
        clear : bool, optional
            Specifies whether to drop the attribute table after attach it
            into the weight table.

        '''
        self._retrieve_('table.attribute',
                        task='ADD', attrtable=attr_tbl,
                        **self.model_weights.to_table_params())

        if clear:
            self._retrieve_('table.droptable',
                            table=attr_tbl)

        print('NOTE: Model attributes attached successfully!')

    def load_weights_attr(self, path):
        '''
        Load the weights attribute form a sashdat file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight attribute table.

        '''
        server_sep = get_server_path_sep(self.conn)
        if file_exist_on_server(self.conn, path):
            if server_sep in path:
                dir_name, file_name = path.rsplit(server_sep, 1)
            else:
                file_name = path
        else:
            raise DLPyError('The specified file does not exist: ' + path)

        try:
            flag, cas_lib_name = check_caslib(self.conn, dir_name)
        except:
            flag = False
            cas_lib_name = random_name('Caslib', 6)
            self._retrieve_('table.addcaslib',
                            name=cas_lib_name, path=dir_name,
                            activeOnAdd=False, dataSource=dict(srcType='DNFS'))

        self._retrieve_('table.loadtable',
                        caslib=cas_lib_name,
                        path=file_name,
                        casout=dict(replace=True,
                                    name=self.model_name + '_weights_attr'))

        self.set_weights_attr(self.model_name + '_weights_attr')

        if not flag:
            self._retrieve_('table.dropcaslib', caslib=cas_lib_name)

    def share_weights(self, layers):
        """
        Share weights between layers

        Parameters
        ----------
        layers : iter-of-dict or dict
            Pass a list of dictionary or a dictionary. Key specifies a layer name.
            Value is the name of layers whose weights will be shared with the layer specified in key, such as
            [dict('conv1_1', ['conv1_2', 'conv1_3', 'conv1_4']), dict('conv2_1', ['conv2_2', 'conv2_3', 'conv2_4'])]


        """
        if not isinstance(layers, list):
            layers = [layers]
        layers_name = [l.name for l in self.layers]
        for layer in layers:
            for anchor, shares in layer.items():
                if anchor not in layers_name:
                    raise DLPyError('{} is not in the model. Please check again.'.format(anchor))
                if isinstance(shares, str):
                    shares = [shares]
                for share in shares:
                    try:
                        idx_share = layers_name.index(share)
                    except ValueError:
                        raise DLPyError('{} is not in the model. Please check again.'.format(anchor))
                    self.layers[idx_share].shared_weights = anchor

    def save_to_astore(self, path=None, layers=None, **kwargs):
        """
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.
        layers : string list, optional
             Specifies the names of the layers to include in the output astore scoring results. This can be used to
             extract the features for given layers.
        """
        self.conn.loadactionset('astore', _messagelevel='error')

        CAS_tbl_name = self.model_name + '_astore'

        self._retrieve_('deeplearn.dlexportmodel',
                        casout=dict(replace=True, name=CAS_tbl_name),
                        initWeights=self.model_weights,
                        modelTable=self.model_table,
                        randomCrop='none',
                        randomFlip='none',
                        randomMutation='none',
                        layers=layers,
                        layerImageType='wide',
                        **kwargs)

        model_astore = self._retrieve_('astore.download',
                                       rstore=CAS_tbl_name)

        file_name = self.model_name + '.astore'
        if path is None:
            path = os.getcwd()

        if not os.path.exists(path):
            raise DLPyError('There seems to be an error while writing the astore file to the client. '
                            'Please check out the path provided. There is also a chance that you are passing '
                            'a server side path while the function expects a client side path')
        else:
            file_name = os.path.join(path, file_name)
            file = open(file_name, 'wb')
            file.write(model_astore['blob'])
            file.close()
            print('NOTE: Model astore file saved successfully.')

    def save_to_table(self, path):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        self.save_to_table_with_caslibify(path)

    def save_to_table_with_caslibify(self, path):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        # import os
        # if path.endswith(os.path.sep):
        #    path = path[:-1]

        with caslibify_context(self.conn, path, task='save') as (caslib, path_remaining):

            _file_name_ = self.model_name.replace(' ', '_')
            _extension_ = '.sashdat'
            model_tbl_file = path_remaining + _file_name_ + _extension_
            weight_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
            attr_tbl_file = path_remaining + _file_name_ + '_weights_attr' + _extension_

            if self.model_table is not None:
                ch = self.conn.table.tableexists(self.model_weights)
                if ch.exists > 0:
                    rt = self._retrieve_('table.save', table=self.model_table, name=model_tbl_file, replace=True,
                                         caslib=caslib)
                    if rt.severity > 1:
                        for msg in rt.messages:
                            print(msg)
                        raise DLPyError('something is wrong while saving the model to a table!')
            if self.model_weights is not None:
                ch = self.conn.table.tableexists(self.model_weights)
                if ch.exists > 0:
                    rt = self._retrieve_('table.save', table=self.model_weights, name=weight_tbl_file,
                                         replace=True, caslib=caslib)
                    if rt.severity > 1:
                        for msg in rt.messages:
                            print(msg)
                        raise DLPyError('something is wrong while saving the model weights to a table!')

                    CAS_tbl_name = random_name('Attr_Tbl')
                    rt = self._retrieve_('table.attribute', task='convert', attrtable=CAS_tbl_name,
                                         **self.model_weights.to_table_params())
                    if rt.severity > 1:
                        for msg in rt.messages:
                            print(msg)
                        raise DLPyError('something is wrong while extracting the model attributes!')

                    rt = self._retrieve_('table.save', table=CAS_tbl_name, name=attr_tbl_file, replace=True,
                                         caslib=caslib)
                    if rt.severity > 1:
                        for msg in rt.messages:
                            print(msg)
                        raise DLPyError('something is wrong while saving the model attributes to a table!')

            print('NOTE: Model table saved successfully.')

    def save_weights_csv(self, path):
        '''
        Save model weights table as csv

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model
            weights csv.

        '''
        weights_table_opts = input_table_check(self.model_weights)
        weights_table_opts.update(**dict(groupBy='_LayerID_',
                                         groupByMode='REDISTRIBUTE',
                                         orderBy='_WeightID_'))
        self.conn.partition(table=weights_table_opts,
                            casout=dict(name=self.model_weights.name,
                                        replace=True))

        with caslibify_context(self.conn, path, task='save') as (caslib, path_remaining):
            _file_name_ = self.model_name.replace(' ', '_')
            _extension_ = '.csv'
            weights_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
            rt = self._retrieve_('table.save', table=weights_table_opts,
                                 name=weights_tbl_file, replace=True, caslib=caslib)
            if rt.severity > 1:
                for msg in rt.messages:
                    print(msg)
                raise DLPyError('something is wrong while saving the the model to a table!')

        print('NOTE: Model weights csv saved successfully.')

    def save_to_onnx(self, path, model_weights=None):
        '''
        Save to ONNX model

        Parameters
        ----------
        path : string
            Specifies the client-side path to save the ONNX model.
        model_weights : string, optional
            Specifies the client-side path of the csv file of the
            model weights table.  If no csv file is specified, the
            weights will be fetched from the CAS server.  This can
            take a long time to complete if the size of model weights
            is large.

        '''

        from .model_conversion.write_onnx_model import sas_to_onnx
        if model_weights is None:
            try:
                self.model_weights.numrows()
            except:
                raise DLPyError('No model weights yet. Please load weights or'
                                ' train the model first.')
            print('NOTE: Model weights will be fetched from server')
            model_weights = self.model_weights
        else:
            print('NOTE: Model weights will be loaded from csv.')
            model_weights = pd.read_csv(model_weights)
        model_table = self.conn.CASTable(**self.model_table)
        onnx_model = sas_to_onnx(layers=self.layers,
                                 model_table=model_table,
                                 model_weights=model_weights)
        file_name = self.model_name + '.onnx'
        if path is None:
            path = os.getcwd()

        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)

        with open(file_name, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print('NOTE: ONNX model file saved successfully.')

    def deploy(self, path, output_format='astore', model_weights=None, layers=None, **kwargs):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the location to store the model files.
            If the output_format is set to castable, then the location has to be on the server-side.
            Otherwise, the location has to be on the client-side.
        output_format : string, optional
            Specifies the format of the deployed model
            Valid Values: astore, castable, or onnx
            Default: astore
        model_weights : string, optional
            Specifies the client-side path to the csv file of the
            model weights table.  Only effective when
            output_format='onnx'.  If no csv file is specified when
            deploying to ONNX, the weights will be fetched from the
            CAS server.  This may take a long time to complete if
            the size of model weights is large.
        layers : string list, optional
             Specifies the names of the layers to include in the output astore scoring results. This can be used to
             extract the features for given layers.

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


        """
        if output_format.lower() == 'astore':
            self.save_to_astore(path=path, layers=layers, **kwargs)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path=path)
        elif output_format.lower() == 'onnx':
            self.save_to_onnx(path, model_weights=model_weights)
        else:
            raise DLPyError('output_format must be "astore", "castable", "table",'
                            'or "onnx"')

    def count_params(self):
        ''' Count the total number of parameters in the model '''
        count = 0
        for layer in self.layers:

            if layer.num_weights is None:
                num_weights = 0
            else:
                num_weights = layer.num_weights

            if layer.num_bias is None:
                num_bias = 0
            else:
                num_bias = layer.num_bias

            count += num_weights + num_bias
        return int(count)

    def create_layer_id_name_mapping(self):
        """
        Create a dictionary which maps layer id to layer name.
        One use case is the model creates weights table given a pre-trained weights and a pre-trained model.
        Example:
            mapper = model.create_layer_id_name_mapping()
            pretrained_weights_table = WeightsTable(conn, weights_tbl_name='my_pretrained_weights_table',
                                                model_tbl_name='my_pretrained_model_table')
            pretrained_weights_table.remap_layer_ids(mapper, casout='new_weights)

        Returns
        -------
        :class:`dict`

        """
        m_frame = self.conn.fetch(dict(name=self.model_name, where='_DLKey1_ eq "layertype"'), to=100000).Fetch
        layer_names = m_frame['_DLKey0_'].values
        layer_ids = m_frame['_DLLayerID_'].values
        return dict(zip(layer_ids, layer_names))


class WeightsTable:
    '''
    WeightsTable
    A weights table builds connection with a deep learning model.
    One use case is a new model setting a pre-trained weights. Since SAS deep learning model loads weights according to
    layer id and the order of layer id might be different between the new built model and the pre-trained model.
    So, the instance of the class can remap layer id and generate a suitable weights table according to layer names.
    Example:
        pretrained_weights_table = WeightsTable(conn, weights_tbl_name='my_pretrained_weights_table',
                                                model_tbl_name='my_pretrained_model_table')
        new_model.set_weights(pretrained_weights_table)

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    weights_tbl_name: string
        Specifies the name of CASTable containing weights of the deep learning model.
    model_tbl_name: string
        Specifies the name of CAS table to store the deep learning model.

    Returns
    -------
    :class:`WeightsTable`

    '''

    def __init__(self, conn, weights_tbl_name, model_tbl_name):
        self.conn = conn
        self._weights_tbl_name = weights_tbl_name
        self._model_tbl_name = model_tbl_name

    @property
    def weights_tbl_name(self):
        return self._weights_tbl_name

    @property
    def model_tbl_name(self):
        return self._model_tbl_name

    @property
    def weights_mapping(self):
        m_frame = self.conn.fetch(dict(name=self.model_tbl_name,
                                       where='_DLKey1_ eq "layertype"'), to=100000).Fetch
        layer_names = m_frame['_DLKey0_'].values
        layer_ids = m_frame['_DLLayerID_'].values
        return dict(zip(layer_names, layer_ids))

    def remap_layer_ids(self, mapper, casout):
        '''
        Remap and generate a new weights table given specified mapper.

        Parameters
        ----------
        mapper : dict
            Specifies the mapper to remap the original weights table. The dictionrary maps layer id to layer name.
            Example:
                {3.0: 'convo.1_3',
                 4.0: 'pool1_3',
                 0.0: 'input_layer_00',
                 6.0: 'pool2_3',
                 2.0: 'input_layer_02',
                 1.0: 'input_layer_01',
                 5.0: 'convo.2_3'}
        casout : string
            Specifies the name of the new weights table.

        '''
        if self.conn.tableexists(casout).exists:
            print('WARNING: The table, {}, has already existed. It will be deleted and recreated.'.format(casout))
            self.conn.droptable(casout)

        tmp_col = random_name('TMPCOL')
        tmp_tbl = random_name('TMPTBL')  # each layer
        tmp_res_tbl = random_name('TMPRESTBL')  # final results
        orig_weights_tbl = self.weights_tbl_name
        if orig_weights_tbl == casout:
            print('WARNING: casout is the same as original weights table name, the original one will be overwritten.')
        old_mapper = self.weights_mapping
        for new_layer_id, layer_name in mapper.items():
            orig_weights_cas_tbl = self.conn.CASTable(orig_weights_tbl)
            try:
                old_layer_id = old_mapper[layer_name]
            except KeyError as k:
                print("WARNING: The layer, {}, is not found in {}.".format(k.args[0], self.model_tbl_name))
                continue
            orig_weights_cas_tbl.append_where('_LayerID_ eq {}'.format(old_layer_id))
            orig_weights_cas_tbl.append_computedvarsprogram('{} = {}'.format(tmp_col, new_layer_id))
            if self.conn.tableexists(tmp_res_tbl).exists:
                self.conn.partition(table=input_table_check(orig_weights_cas_tbl),
                                    casout=dict(name=tmp_tbl, replace=True))
                tmp_res_cas_tbl = self.conn.CASTable(tmp_res_tbl)
                tmp_cas_tbl = self.conn.CASTable(tmp_tbl)
                tmp_res_cas_tbl.append(tmp_cas_tbl, casout=dict(name=tmp_res_tbl, replace=True))
            else:
                self.conn.partition(table=input_table_check(orig_weights_cas_tbl),
                                    casout=dict(name=tmp_res_tbl, replace=True))

        self.conn.altertable(name=tmp_res_tbl, rename=casout, drop='_LayerID_')
        self.conn.altertable(name=casout, columns=[dict(name=tmp_col, rename='_LayerID_')],
                             columnOrder=['_LayerID_', '_WeightID_', '_Weight_'])
        with sw.option_context(print_messages=False):
            self.conn.droptable(tmp_tbl)
            self.conn.droptable(tmp_res_tbl)


def layer_to_node(layer):
    '''
    Convert layer configuration to a node in the model graph

    Parameters
    ----------
    layer : Layer object
        Specifies the layer to be converted.

    Returns
    -------
    :class:`dict`
        Options that can be passed to graph configuration.

    '''
    if layer.type == 'recurrent':
        label = '%s(%s)' % (layer.name, layer.type)
    else:
        if layer.kernel_size:
            label = '%s %s(%s)' % ('x'.join('%s' % x for x in layer.kernel_size), layer.name, layer.type)
        elif layer.output_size:
            if not isinstance(layer.output_size, collections.Iterable):
                label = '%s %s(%s)' % (layer.output_size, layer.name, layer.type)
            else:
                label = '%s %s(%s)' % ('x'.join('%s' % x for x in layer.output_size), layer.name, layer.type)
        else:
            label = '%s(%s)' % (layer.name, layer.type)

    if isinstance(layer.color_code, (list, tuple)):
        fg = layer.color_code[0]
        bg = layer.color_code[1]
    else:
        fg = layer.color_code[:7]
        bg = layer.color_code

    return dict(name=layer.name, label=' %s ' % label,
                fillcolor=bg, color=fg, margin='0.2,0.0', height='0.3')


def layer_to_edge(layer):
    '''
    Convert layer to layer connection to an edge in the model graph

    Parameters
    ----------
    layer : Layer object
        Specifies the layer to be converted.

    Returns
    -------
    :class:`dict`
        Options that can be passed to graph configuration.

    '''
    gv_params = []
    for item in layer.src_layers:
        label = ''
        if layer.type is not 'input':
            if isinstance(item.output_size, (tuple, list)):
                label = ' %s ' % ' x '.join('%s' % x for x in item.output_size)
            else:
                label = ' %s ' % item.output_size
        gv_params.append(dict(label=label, tail_name='{}'.format(item.name),
                              head_name='{}'.format(layer.name)))

    if layer.type == 'recurrent':
        gv_params.append(dict(label='', tail_name='{}'.format(layer.name),
                              head_name='{}'.format(layer.name)))
    return gv_params


def model_to_graph(model):
    '''
    Convert model configuration to a graph

    Parameters
    ----------
    model : Model object
        Specifies the model to be converted.

    Returns
    -------
    :class:`graphviz.dot.Digraph`

    '''
    import graphviz as gv

    model_graph = gv.Digraph(name=model.model_name,
                             node_attr=dict(shape='record', style='filled', fontname='helvetica'),
                             edge_attr=dict(fontname='helvetica', fontsize='10'))
    # can be added later for adjusting figure size.
    # fixedsize='True', width = '4', height = '1'))

    #   model_graph.attr(label=r'DAG for {}:'.format(model.model_name),
    #                    labelloc='top', labeljust='left')
    #   model_graph.attr(fontsize='16')

    for layer in model.layers:
        if layer.type == 'input':
            model_graph.node(**layer_to_node(layer))
        else:
            model_graph.node(**layer_to_node(layer))
            for gv_param in layer_to_edge(layer):
                model_graph.edge(color='#5677F3', **gv_param)

    return model_graph


def get_num_configs(keys, layer_type_prefix, layer_table):
    '''
    Extract the numerical options from the model table

    Parameters
    ----------
    keys : list-of-strings
        Specifies the list of numerical variables
    layer_type_prefix : string
        Specifies the prefix of the options in the model table
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' +
                key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def get_str_configs(keys, layer_type_prefix, layer_table):
    '''
    Extract the str options from the model table

    Parameters
    ----------
    keys : list-of-strings
        Specifies the list of str variables.
    layer_type_prefix : string
        Specifies the prefix of the options in the model table.
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition.

    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLChrVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' +
                key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def extract_input_layer(layer_table):
    '''
    Extract layer configuration from an input layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n_channels', 'width', 'height', 'dropout', 'scale']
    input_layer_config = dict()
    input_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    input_layer_config.update(get_num_configs(num_keys, 'inputopts', layer_table))

    input_layer_config['offsets'] = []
    try:
        input_layer_config['offsets'].append(
            int(layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                          'inputopts.offsets'].tolist()[0]))
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.0'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.1'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.offsets.2'].tolist()[0])
    except IndexError:
        pass

    input_layer_config['norm_stds'] = []
    try:
        input_layer_config['norm_stds'].append(
            int(layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                          'inputopts.normstds'].tolist()[0]))
    except IndexError:
        pass
    try:
        input_layer_config['norm_stds'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.normstds.0'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['norm_stds'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.normstds.1'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['norm_stds'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'inputopts.normstds.2'].tolist()[0])
    except IndexError:
        pass

    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] ==
                                 'inputopts.crop'].tolist()[0] == 'No cropping':
        input_layer_config['random_crop'] = 'none'
    else:
        input_layer_config['random_crop'] = 'unique'

    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] ==
                                 'inputopts.flip'].tolist()[0] == 'No flipping':
        input_layer_config['random_flip'] = 'none'
    # else:
    #     input_layer_config['random_flip']='hv'

    layer = InputLayer(**input_layer_config)
    return layer


def extract_conv_layer(layer_table):
    '''
    Extract layer configuration from a convolution layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    conv_layer_config = dict()
    conv_layer_config.update(get_num_configs(num_keys, 'convopts', layer_table))
    conv_layer_config.update(get_str_configs(str_keys, 'convopts', layer_table))
    conv_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in conv_layer_config:
        conv_layer_config['truncation_factor'] = conv_layer_config['trunc_fact']
        del conv_layer_config['trunc_fact']
    if conv_layer_config.get('act').lower() == 'leaky activation function':
        conv_layer_config['act'] = 'Leaky'

    dl_numval = layer_table['_DLNumVal_']
    if dl_numval[layer_table['_DLKey1_'] == 'convopts.no_bias'].any():
        conv_layer_config['include_bias'] = False
    else:
        conv_layer_config['include_bias'] = True

    # pad_top and pad_left are added after vb015
    if 'convopts.pad_left' in layer_table['_DLKey1_'].values and 'convopts.pad_top' in layer_table['_DLKey1_'].values:
        padding_width = dl_numval[layer_table['_DLKey1_'] == 'convopts.pad_left'].tolist()[0]
        padding_height = dl_numval[layer_table['_DLKey1_'] == 'convopts.pad_top'].tolist()[0]
        if padding_width != -1:
            conv_layer_config['padding_width'] = padding_width
        if padding_height != -1:
            conv_layer_config['padding_height'] = padding_height

    layer = Conv2d(**conv_layer_config)
    return layer


def extract_pooling_layer(layer_table):
    '''
    Extract layer configuration from a pooling layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['width', 'height', 'stride']
    str_keys = ['act', 'poolingtype']

    pool_layer_config = dict()
    pool_layer_config.update(get_num_configs(num_keys, 'poolingopts', layer_table))
    pool_layer_config.update(get_str_configs(str_keys, 'poolingopts', layer_table))

    pool_layer_config['pool'] = pool_layer_config['poolingtype'].lower().split(' ')[0]
    del pool_layer_config['poolingtype']
    pool_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    # pad_top and pad_left are added after vb015
    if 'poolingopts.pad_left' in layer_table['_DLKey1_'].values and 'poolingopts.pad_top' in layer_table[
        '_DLKey1_'].values:
        padding_width = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'poolingopts.pad_left'].tolist()[0]
        padding_height = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'poolingopts.pad_top'].tolist()[0]
        if padding_width != -1:
            pool_layer_config['padding_width'] = padding_width
        if padding_height != -1:
            pool_layer_config['padding_height'] = padding_height

    layer = Pooling(**pool_layer_config)
    return layer


def extract_batchnorm_layer(layer_table):
    '''
    Extract layer configuration from a batch normalization layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    bn_layer_config = dict()
    bn_layer_config.update(get_str_configs(['act'], 'bnopts', layer_table))
    bn_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    if bn_layer_config.get('act').lower() == 'leaky activation function':
        bn_layer_config['act'] = 'Leaky'

    layer = BN(**bn_layer_config)
    return layer


def extract_residual_layer(layer_table):
    '''
    Extract layer configuration from a residual layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    res_layer_config = dict()

    res_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    res_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    # manually correct table content as a valid option
    if res_layer_config['act'] == 'Automatic':
        res_layer_config['act'] = 'AUTO'

    layer = Res(**res_layer_config)
    return layer


def extract_concatenate_layer(layer_table):
    '''
    Extract layer configuration from a concatenate layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    concat_layer_config = dict()

    concat_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    concat_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Concat(**concat_layer_config)
    return layer


def extract_detection_layer(layer_table):
    '''
    Extract layer configuration from a detection layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    num_keys = ['num_to_force_coord', 'softmax_for_class_prob', 'detection_threshold',
                'force_coord_scale', 'prediction_not_a_object_scale', 'coord_scale', 'predictions_per_grid',
                'object_scale', 'iou_threshold', 'class_scale', 'max_label_per_image', 'max_boxes', 'match_anchor_size',
                'do_sqrt', 'class_number', 'coord_type', 'grid_number']
    str_keys = ['act', 'init']

    detection_layer_config = dict()
    for key in num_keys:
        try:
            detection_layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == 'detectionopts.' + underscore_to_camelcase(key)].tolist()[0]
        except IndexError:
            pass

    for key in str_keys:
        try:
            detection_layer_config[key] = layer_table['_DLChrVal_'][
                layer_table['_DLKey1_'] == 'detectionopts.' + underscore_to_camelcase(key)].tolist()[0]
        except IndexError:
            pass

    detection_layer_config['detection_model_type'] = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                                               'detectionopts.yoloVersion'].tolist()[0]

    predictions_per_grid = detection_layer_config['predictions_per_grid']
    detection_layer_config['anchors'] = []
    for i in range(int(predictions_per_grid * 2)):
        detection_layer_config['anchors'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'detectionopts.anchors.{}'.format(i)].tolist()[0])

    detection_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Detection(**detection_layer_config)
    return layer


def extract_fc_layer(layer_table):
    '''
    Extract layer configuration from a fully connected layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    fc_layer_config = dict()
    fc_layer_config.update(get_num_configs(num_keys, 'fcopts', layer_table))
    fc_layer_config.update(get_str_configs(str_keys, 'fcopts', layer_table))
    fc_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'fcopts.no_bias'].any():
        fc_layer_config['include_bias'] = False
    else:
        fc_layer_config['include_bias'] = True

    if 'trunc_fact' in fc_layer_config:
        fc_layer_config['truncation_factor'] = fc_layer_config['trunc_fact']
        del fc_layer_config['trunc_fact']
    if fc_layer_config.get('act').lower() == 'leaky activation function':
        fc_layer_config['act'] = 'Leaky'

    layer = Dense(**fc_layer_config)
    return layer


def extract_recurrent_layer(layer_table):
    '''
    Extract layer configuration from a recurrent layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'std', 'mean', 'max_output_length',
                'dropout', 'reversed', 'trunc_fact']
    str_keys = ['act', 'init', 'rnn_type', 'rnn_outputtype']

    recurrent_layer_config = dict()
    recurrent_layer_config.update(get_num_configs(num_keys, 'rnnopts', layer_table))
    recurrent_layer_config.update(get_str_configs(str_keys, 'rnnopts', layer_table))
    recurrent_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in recurrent_layer_config.keys():
        recurrent_layer_config['truncation_factor'] = recurrent_layer_config['trunc_fact']
        del recurrent_layer_config['trunc_fact']

    if 'reversed' in recurrent_layer_config.keys():
        if recurrent_layer_config['reversed'] > 0:
            recurrent_layer_config['reversed_'] = True
        else:
            recurrent_layer_config['reversed_'] = False
        del recurrent_layer_config['reversed']
    else:
        recurrent_layer_config['reversed_'] = False

    if 'rnn_type' in recurrent_layer_config.keys():
        if 'Long' in recurrent_layer_config['rnn_type']:
            recurrent_layer_config['rnn_type'] = 'LSTM'
        elif 'Gated' in recurrent_layer_config['rnn_type']:
            recurrent_layer_config['rnn_type'] = 'GRU'
        else:
            recurrent_layer_config['rnn_type'] = 'RNN'
    else:
        recurrent_layer_config['rnn_type'] = 'RNN'

    if 'act' in recurrent_layer_config.keys():
        if 'Hyperbolic' in recurrent_layer_config['act']:
            recurrent_layer_config['act'] = 'TANH'
        elif recurrent_layer_config['act'] == 'Automatic':
            recurrent_layer_config['act'] = 'AUTO'
        elif recurrent_layer_config['act'] == 'Identity':
            recurrent_layer_config['act'] = 'IDENTITY'
        elif recurrent_layer_config['act'] == 'Logistic':
            recurrent_layer_config['act'] = 'LOGISTIC'
        elif recurrent_layer_config['act'] == 'Sigmoid':
            recurrent_layer_config['act'] = 'SIGMOID'
        else:
            recurrent_layer_config['act'] = 'AUTO'
    else:
        recurrent_layer_config['act'] = 'AUTO'

    if 'rnn_outputtype' in recurrent_layer_config.keys():
        if 'arbitrary' in recurrent_layer_config['rnn_outputtype']:
            recurrent_layer_config['output_type'] = 'ARBITRARYLENGTH'
        elif 'fixed-length' in recurrent_layer_config['rnn_outputtype']:
            recurrent_layer_config['output_type'] = 'ENCODING'
        else:
            recurrent_layer_config['output_type'] = 'SAMELENGTH'
        del recurrent_layer_config['rnn_outputtype']
    else:
        recurrent_layer_config['output_type'] = 'SAMELENGTH'

    layer = Recurrent(**recurrent_layer_config)
    return layer


def extract_output_layer(layer_table):
    '''
    Extract layer configuration from an output layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    output_layer_config = dict()
    output_layer_config.update(get_num_configs(num_keys, 'outputopts', layer_table))
    output_layer_config.update(get_str_configs(str_keys, 'outputopts', layer_table))
    output_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'outputopts.no_bias'].any():
        output_layer_config['include_bias'] = False
    else:
        output_layer_config['include_bias'] = True

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'outputopts.noFullConnect'].any():
        output_layer_config['full_connect'] = False
    else:
        output_layer_config['full_connect'] = True

    if 'trunc_fact' in output_layer_config:
        output_layer_config['truncation_factor'] = output_layer_config['trunc_fact']
        del output_layer_config['trunc_fact']

    layer = OutputLayer(**output_layer_config)
    return layer


def extract_scale_layer(layer_table):
    '''
    Extract layer configuration from a scale layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    num_keys = ['scale']
    str_keys = ['act']
    scale_layer_config = dict()
    scale_layer_config.update(get_num_configs(num_keys, 'scaleopts', layer_table))
    scale_layer_config.update(get_str_configs(str_keys, 'scaleopts', layer_table))
    scale_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    layer = Scale(**scale_layer_config)
    return layer


def extract_keypoints_layer(layer_table):
    '''
    Extract layer configuration from a keypoints layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    num_keys = ['n', 'std', 'mean', 'init_bias', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'error']
    keypoints_layer_config = dict()
    keypoints_layer_config.update(get_num_configs(num_keys, 'keypointsopts', layer_table))
    keypoints_layer_config.update(get_str_configs(str_keys, 'keypointsopts', layer_table))
    keypoints_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    layer = Keypoints(**keypoints_layer_config)

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'keypointsopts.no_bias'].any():
        keypoints_layer_config['include_bias'] = False
    else:
        keypoints_layer_config['include_bias'] = True

    if 'trunc_fact' in keypoints_layer_config:
        keypoints_layer_config['truncation_factor'] = keypoints_layer_config['trunc_fact']
        del keypoints_layer_config['trunc_fact']
    return layer


def extract_reshape_layer(layer_table):
    '''
    Extract layer configuration from a reshape layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    num_keys = ['n', 'width', 'height', 'depth']
    str_keys = ['act']
    reshape_layer_config = dict()
    reshape_layer_config.update(get_num_configs(num_keys, 'reshapeopts', layer_table))
    reshape_layer_config.update(get_str_configs(str_keys, 'reshapeopts', layer_table))
    if reshape_layer_config['act'] == 'Automatic':
        reshape_layer_config['act'] = 'AUTO'
    reshape_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    layer = Reshape(**reshape_layer_config)
    return layer


def extract_groupconv_layer(layer_table):
    '''
    Extract layer configuration from a group convolution layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact', 'n_groups']
    str_keys = ['act', 'init']

    grpconv_layer_config = dict()
    grpconv_layer_config.update(get_num_configs(num_keys, 'groupconvopts', layer_table))
    grpconv_layer_config.update(get_str_configs(str_keys, 'groupconvopts', layer_table))
    grpconv_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in grpconv_layer_config:
        grpconv_layer_config['truncation_factor'] = grpconv_layer_config['trunc_fact']
        del grpconv_layer_config['trunc_fact']
    if grpconv_layer_config.get('act').lower() == 'leaky activation function':
        grpconv_layer_config['act'] = 'Leaky'

    dl_numval = layer_table['_DLNumVal_']
    if dl_numval[layer_table['_DLKey1_'] == 'groupconvopts.no_bias'].any():
        grpconv_layer_config['include_bias'] = False
    else:
        grpconv_layer_config['include_bias'] = True

    padding_width = dl_numval[layer_table['_DLKey1_'] == 'groupconvopts.pad_left'].tolist()[0]
    padding_height = dl_numval[layer_table['_DLKey1_'] == 'groupconvopts.pad_top'].tolist()[0]
    if padding_width != -1:
        grpconv_layer_config['padding_width'] = padding_width
    if padding_height != -1:
        grpconv_layer_config['padding_height'] = padding_height

    layer = GroupConv2d(**grpconv_layer_config)
    return layer


def extract_conv2dtranspose_layer(layer_table):
    '''
    Extract layer configuration from a Conv2DTranspose layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact',
                'output_padding_height', 'output_padding_width']
    str_keys = ['act', 'init']

    conv2dtranspose_layer_config = dict()
    conv2dtranspose_layer_config.update(get_num_configs(num_keys, 'transposeconvopts', layer_table))
    conv2dtranspose_layer_config.update(get_str_configs(str_keys, 'transposeconvopts', layer_table))
    conv2dtranspose_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in conv2dtranspose_layer_config:
        conv2dtranspose_layer_config['truncation_factor'] = conv2dtranspose_layer_config['trunc_fact']
        del conv2dtranspose_layer_config['trunc_fact']
    if conv2dtranspose_layer_config.get('act').lower() == 'leaky activation function':
        conv2dtranspose_layer_config['act'] = 'Leaky'

    dl_numval = layer_table['_DLNumVal_']
    if dl_numval[layer_table['_DLKey1_'] == 'transposeconvopts.no_bias'].any():
        conv2dtranspose_layer_config['include_bias'] = False
    else:
        conv2dtranspose_layer_config['include_bias'] = True

    padding_width = dl_numval[layer_table['_DLKey1_'] == 'transposeconvopts.pad_left'].tolist()[0]
    padding_height = dl_numval[layer_table['_DLKey1_'] == 'transposeconvopts.pad_top'].tolist()[0]
    if padding_width != -1:
        conv2dtranspose_layer_config['padding_width'] = padding_width
    if padding_height != -1:
        conv2dtranspose_layer_config['padding_height'] = padding_height

    layer = Conv2DTranspose(**conv2dtranspose_layer_config)
    return layer


def extract_channelshuffle_layer(layer_table):
    '''
    Extract layer configuration from a channel shuffle layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['scale', 'n_groups']
    str_keys = ['init']

    channel_shuffle_layer_config = dict()
    channel_shuffle_layer_config.update(get_num_configs(num_keys, 'shuffleopts', layer_table))
    channel_shuffle_layer_config.update(get_str_configs(str_keys, 'shuffleopts', layer_table))
    channel_shuffle_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = ChannelShuffle(**channel_shuffle_layer_config)
    return layer


def extract_fcmp_layer(layer_table):
    '''
    Extract layer configuration from a FCMP layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['width', 'height', 'depth', 'n_weights']

    fcmp_layer_config = dict()
    fcmp_layer_config.update(get_num_configs(num_keys, 'fcmpopts', layer_table))
    # forward and backward function
    fcmp_layer_config['forward_func'] = layer_table['_DLChrVal_'][
        layer_table['_DLKey1_'] == 'fcmpopts.fcmp'].tolist()[0]
    fcmp_layer_config['backward_func'] = layer_table['_DLChrVal_'][
        layer_table['_DLKey1_'] == 'fcmpopts.fcmpder'].tolist()[0]

    fcmp_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = FCMPLayer(**fcmp_layer_config)
    return layer


def extract_segmentation_layer(layer_table):
    '''
    Extract layer configuration from a Segmentation layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    str_keys = ['act', 'error']

    segmentation_layer_config = dict()
    segmentation_layer_config['target_scale'] = layer_table['_DLNumVal_'][
        layer_table['_DLKey1_'] == 'segmentationopts.targetScale'].tolist()[0]

    segmentation_layer_config.update(get_str_configs(str_keys, 'segmentationopts', layer_table))
    # correct act if it is Automatic
    if segmentation_layer_config['act'] == 'Automatic':
        segmentation_layer_config['act'] = 'AUTO'
    # correct error if it is Automatic
    if segmentation_layer_config['error'] == 'Automatic':
        segmentation_layer_config['error'] = 'AUTO'

    segmentation_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Segmentation(**segmentation_layer_config)
    return layer


def extract_embeddingloss_layer(layer_table):
    '''
    Extract layer configuration from a Embedding Loss layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''

    embeddingloss_layer_config = dict()
    embeddingloss_layer_config['margin'] = layer_table['_DLNumVal_'][
        layer_table['_DLKey1_'] == 'clossopts.margin'].tolist()[0]

    embeddingloss_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = EmbeddingLoss(**embeddingloss_layer_config)
    return layer


def extract_cluster_layer(layer_table):
    '''
    Extract layer configuration from a Clustering layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    cluster_layer_config = dict()
    cluster_layer_config['n_clusters'] = layer_table['_DLNumVal_'][
        layer_table['_DLKey1_'] == 'clusteropts.nClusters'].tolist()[0]
    cluster_layer_config['alpha'] = layer_table['_DLNumVal_'][
        layer_table['_DLKey1_'] == 'clusteropts.alpha'].tolist()[0]

    cluster_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Clustering(**cluster_layer_config)
    return layer


def extract_rpn_layer(layer_table):
    '''
    Extract layer configuration from a Region proposal layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['base_anchor_size', 'max_label_per_image', 'roi_train_sample_num', 'do_RPN_only',
                'proposed_roi_num_train', 'proposed_roi_num_score', 'anchor_num_to_sample']
    if __dev__:
        num_keys += ['preNmsTopNScore', 'preNmsTopNTrain', 'preNmsTopNTrain', 'preNmsTopNScore']
    str_key = 'act'

    rpn_layer_config = dict()
    for key in num_keys:
        try:
            rpn_layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == 'dlregionproposalopts.' + underscore_to_camelcase(key)].tolist()[0]
        except IndexError:
            pass

    rpn_layer_config[str_key] = layer_table['_DLChrVal_'][
        layer_table['_DLKey1_'] == 'dlregionproposalopts.' + underscore_to_camelcase(str_key)].tolist()[0]

    num_scale = layer_table[layer_table['_DLChrVal_'] == 'anchorScale'].shape[0]
    num_ratio = layer_table[layer_table['_DLChrVal_'] == 'anchorRatio'].shape[0]
    rpn_layer_config['anchor_scale'] = []
    rpn_layer_config['anchor_ratio'] = []

    for i in range(num_scale):
        rpn_layer_config['anchor_scale'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'dlregionproposalopts.anchorScale.{}'.format(i)].tolist()[0])

    for i in range(num_ratio):
        rpn_layer_config['anchor_ratio'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                      'dlregionproposalopts.anchorRatio.{}'.format(i)].tolist()[0])

    rpn_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = RegionProposal(**rpn_layer_config)
    return layer


def extract_roipooling_layer(layer_table):
    '''
    Extract layer configuration from a Region pooling layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['output_height', 'spatial_scale', 'output_width']
    str_keys = ['act']

    roipooling_layer_config = dict()
    for key in num_keys:
        try:
            roipooling_layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == 'dlroipoolingopts.' + underscore_to_camelcase(key)].tolist()[0]
        except IndexError:
            pass
    roipooling_layer_config.update(get_str_configs(str_keys, 'dlroipoolingopts', layer_table))

    roipooling_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = ROIPooling(**roipooling_layer_config)
    return layer


def extract_fastrcnn_layer(layer_table):
    '''
    Extract layer configuration from a Fast RCNN layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['class_number', 'max_label_per_image', 'nms_iou_threshold', 'max_object_num', 'detection_threshold']

    rpn_layer_config = dict()
    for key in num_keys:
        try:
            rpn_layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == 'dlfastrcnnopts.' + underscore_to_camelcase(key)].tolist()[0]
        except IndexError:
            pass

    rpn_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = FastRCNN(**rpn_layer_config)
    return layer


def extract_layernorm_layer(layer_table):
    '''
    Extract layer configuration from a layer normalization layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['dropout', 'epsilon']
    str_keys = ['act']

    ln_layer_config = dict()
    ln_layer_config.update(get_num_configs(num_keys, 'dllayernormopts', layer_table))
    ln_layer_config.update(get_str_configs(str_keys, 'dllayernormopts', layer_table))
    ln_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = LayerNormalization(**ln_layer_config)
    return layer


def extract_mhattention_layer(layer_table):
    '''
    Extract layer configuration from a multi-head attention layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    num_keys = ['n', 'n_attn_heads', 'dropout', 'attn_dropout', 'init', 'std', 'mean',
                'truncation_factor', 'trunc_fact']
    str_keys = ['act', 'init', 'include_bias', 'mask']

    mha_layer_config = dict()
    mha_layer_config.update(get_num_configs(num_keys, 'dlmhattentionopts', layer_table))
    mha_layer_config.update(get_str_configs(str_keys, 'dlmhattentionopts', layer_table))
    mha_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in mha_layer_config:
        mha_layer_config['truncation_factor'] = mha_layer_config['trunc_fact']
        del mha_layer_config['trunc_fact']

    layer = MultiHeadAttention(**mha_layer_config)
    return layer


def extract_survival_layer(layer_table):
    '''
    Extract layer configuration from a survival layer table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    survival_layer_config = dict()

    survival_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Survival(**survival_layer_config)
    return layer


def extract_split_layer(layer_table):
    '''
    Extract layer configuration from a Split table

    Parameters
    ----------
    layer_table : table
        Specifies the selection of table containing the information
        for the layer.

    Returns
    -------
    :class:`dict`
        Options that can be passed to layer definition

    '''
    split_layer_config = dict()

    split_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    # one workaround to get number of destination layers.
    split_layer_config['n_destination_layers'] = layer_table[
        layer_table['_DLKey1_'].str.startswith('srclayers', na=False)].shape[0]

    layer = Split(**split_layer_config)
    return layer

