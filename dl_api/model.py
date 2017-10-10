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

"""
Model object for deep learning.
"""

from .utils import random_name
from .utils import input_table_check
from .utils import image_blocksize
from .layers import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


class Model:
    '''
    Model

    Parameters:

    ----------
    conn :
        Specifies the CAS connection.
    model_name : string
        Specifies the name of the deep learning model.
    model_weights : string, dictionary or CAS table, optional
        Specifies the weights of the deep learning model.
        If not specified, random initial will be used.
        Default : None

    Returns

    -------
    A deep learning model objects.
    '''

    @classmethod
    def from_table(cls, model_table):

        model = cls(conn=model_table.get_connection())
        model_name = model.retrieve(_name_='table.fetch',
                                    table=dict(where='_DLKey1_= "modeltype"',
                                               **model_table.to_table_params())).Fetch['_DLKey0_'][0]

        print('NOTE: Model table is attached successfully!\n'
              'NOTE: Model is named to "{}" according to the model name in the table.'.format(model_name))
        model.model_name = model_name
        model.model_weights = model.conn.CASTable('{}_weights'.format(model_name))

        model_table = model_table.to_frame()
        for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
            layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
            layertype = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'layertype'].tolist()[0]
            if layertype == 1:
                model.layers.append(extract_input_layer(layer_table=layer_table))
            elif layertype == 2:
                model.layers.append(extract_conv_layer(layer_table=layer_table))
            elif layertype == 3:
                model.layers.append(extract_pooling_layer(layer_table=layer_table))
            elif layertype == 4:
                model.layers.append(extract_fc_layer(layer_table=layer_table))
            elif layertype == 5:
                model.layers.append(extract_output_layer(layer_table=layer_table))
            elif layertype == 8:
                model.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layertype == 9:
                model.layers.append(extract_residual_layer(layer_table=layer_table))
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
        for layer in model.layers:
            layer.summary()

            # Check if weight table is in the same path
        return model

    @classmethod
    def from_sashdat(cls, conn, path):
        model = Model(conn)
        model.load(path=path)
        return model

    def __init__(self, conn, model_name=None, model_weights=None):

        if not conn.queryactionset('deepLearn')['deepLearn']:
            conn.loadactionset(actionSet='deepLearn', _messagelevel='error')

            # self.table = conn.CASTable(model_name)
        self.conn = conn
        if model_name is None:
            self.model_name = random_name('Model', 6)
        elif type(model_name) is not str:
            raise TypeError('model_name has to be a string type.')
        else:
            self.model_name = model_name

        if model_weights is None:
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))
        else:
            self.set_weights(model_weights)
        self.layers = []
        self.valid_res = None
        self.feature_maps = None
        self.valid_conf_mat = None
        self.valid_score = None
        self.n_epochs = 0
        self.training_history = None

    def retrieve(self, message_level='error', **kwargs):
        return self.conn.retrieve(_messagelevel=message_level, **kwargs)

    def load(self, path):
        '''
        Function to load the deep learning model architecture from existing table.

        Parameters:

        ----------
        path: str
            Specifies the full path of the table.
            Note: the path need to be in Linux path format.
        '''

        dir_name, file_name = path.rsplit('/', 1)

        cas_lib_name = random_name('Caslib', 6)
        self.retrieve(_name_='addcaslib',
                      name=cas_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        self.retrieve(_name_='table.loadtable',
                      caslib=cas_lib_name,
                      path=file_name,
                      casout=dict(replace=True, name=self.model_name))

        model_name = self.retrieve(_name_='table.fetch',
                                   table=dict(name=self.model_name,
                                              where='_DLKey1_= "modeltype"')).Fetch['_DLKey0_'][0]

        if model_name.lower() != self.model_name.lower():
            self.retrieve(_name_='table.partition', casout=dict(replace=True, name=model_name),
                          table=self.model_name)

            self.retrieve(_name_='table.droptable',
                          table=self.model_name)

            print('NOTE: Model table is loaded successfully!\n'
                  'NOTE: Model is renamed to "{}" according to the model name in the table.'.format(model_name))
            self.model_name = model_name
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))

        model_table = self.conn.CASTable(self.model_name).to_frame()
        for layer_id in range(int(model_table['_DLLayerID_'].max()) + 1):
            layer_table = model_table[model_table['_DLLayerID_'] == layer_id]
            layertype = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'layertype'].tolist()[0]
            if layertype == 1:
                self.layers.append(extract_input_layer(layer_table=layer_table))
            elif layertype == 2:
                self.layers.append(extract_conv_layer(layer_table=layer_table))
            elif layertype == 3:
                self.layers.append(extract_pooling_layer(layer_table=layer_table))
            elif layertype == 4:
                self.layers.append(extract_fc_layer(layer_table=layer_table))
            elif layertype == 5:
                self.layers.append(extract_output_layer(layer_table=layer_table))
            elif layertype == 8:
                self.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layertype == 9:
                self.layers.append(extract_residual_layer(layer_table=layer_table))
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
        for layer in self.layers:
            layer.summary()

        # Check if weight table is in the same path
        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(self.retrieve(_name_='table.fileinfo',
                                              caslib=cas_lib_name, includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_weights' + _extension_) in _file_name_list_:
            print('NOTE: ' + _file_name_ + '_weights' + _extension_ + ' is used as model weigths.')

            self.retrieve(_name_='table.loadtable',
                          caslib=cas_lib_name,
                          path=_file_name_ + '_weights' + _extension_,
                          casout=dict(replace=True, name=self.model_name + '_weights'))
            self.set_weights(self.model_name + '_weights')

            if (_file_name_ + '_weights_attr' + _extension_) in _file_name_list_:
                print('NOTE: ' + _file_name_ + '_weights_attr' + _extension_ + ' is used as weigths attribute.')
                self.retrieve(_name_='table.loadtable',
                              caslib=cas_lib_name,
                              path=_file_name_ + '_weights_attr' + _extension_,
                              casout=dict(replace=True, name=self.model_name + '_weights_attr'))
                self.set_weights_attr(self.model_name + '_weights_attr')

        self.retrieve(_name_='dropcaslib', caslib=cas_lib_name)

    def set_weights(self, weight_tbl):

        '''
        Function to assign the weight to the model.


        Parameters:

        ----------
        weight_tbl : A CAS table object, a string specifies the name of the CAS table,
                   a dictionary specifies the CAS table.
            Specifies the weights for the model.

        '''

        weight_tbl = input_table_check(weight_tbl)
        weight_name = self.model_name + '_weights'

        if weight_tbl['name'].lower() != weight_name.lower():
            self.retrieve(_name_='table.partition',
                          casout=dict(replace=True, name=self.model_name + '_weights'),
                          table=weight_tbl)

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')
        print('NOTE: Model weights attached successfully!')

    def load_weights(self, path, kwarg):

        '''
        Function to load the weights form a file.


        Parameters:

        ----------
        path : str
            Specifies the directory of the file that store the weight table.

        '''

        dir_name, file_name = path.rsplit('/', 1)
        if file_name.lower().endswith('.sashdat'):
            self.load_weights_from_table(path)
        if file_name.lower().endswith('.h5'):
            self.load_weights_from_HDF5(path, **kwarg)

    def load_weights_from_HDF5(self, path, kwarg):
        '''
        Function to load the model weights from a HDF5 file.
        Parameters:

        ----------
        path : str
            Specifies the directory of the HDF5 file that store the weight table.
        '''
        self.retrieve(_name_='dlimportmodelweights', model=self.model_name,
                      modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                      formatType="HDF5", weightFilePath=path, **kwarg)

    def load_weights_from_table(self, path):

        '''
        Function to load the weights form a file.


        Parameters:

        ----------
        path : str
            Specifies the directory of the file that store the weight table.

        '''

        dir_name, file_name = path.rsplit('/', 1)

        cas_lib_name = random_name('Caslib', 6)
        self.retrieve(_name_='addcaslib',
                      name=cas_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        self.retrieve(_name_='table.loadtable',
                      caslib=cas_lib_name,
                      path=file_name,
                      casout=dict(replace=True, name=self.model_name + '_weights'))

        self.set_weights(self.model_name + '_weights')

        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(
            self.retrieve(_name_='table.fileinfo', caslib=cas_lib_name, includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_attr' + _extension_) in _file_name_list_:
            print('NOTE: ' + _file_name_ + '_attr' + _extension_ + ' is used as weigths attribute.')
            self.retrieve(_name_='table.loadtable',
                          caslib=cas_lib_name,
                          path=_file_name_ + '_attr' + _extension_,
                          casout=dict(replace=True, name=self.model_name + '_weights_attr'))

            self.set_weights_attr(self.model_name + '_weights_attr')

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')

        self.retrieve(_name_='dropcaslib',
                      caslib=cas_lib_name)

    def set_weights_attr(self, attr_tbl, clear=True):

        '''
        Function to attach the weights attribute.

        Parameters:

        ----------
        attr_tbl : castable parameter
            Specifies the weights attribute table.
        clear : boolean, optional
            Specifies whether to drop the attribute table after attach it into the weight table.

        '''

        self.retrieve(_name_='table.attribute',
                      task='ADD', attrtable=attr_tbl,
                      **self.model_weights.to_table_params())

        if clear:
            self.retrieve(_name_='table.droptable',
                          table=attr_tbl)

        print('NOTE: Model attributes attached successfully!')

    def load_weights_attr(self, path):

        '''
        Function to load the weights attribute form a file.


        Parameters:

        ----------
        path : str
            Specifies the directory of the file that store the weight attribute table.

        '''

        dir_name, file_name = path.rsplit('/', 1)

        cas_lib_name = random_name('Caslib', 6)
        self.retrieve(_name_='addcaslib',
                      name=cas_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        self.retrieve(_name_='table.loadtable',
                      caslib=cas_lib_name,
                      path=file_name,
                      casout=dict(replace=True, name=self.model_name + '_weights_attr'))

        self.set_weights_attr(self.model_name + '_weights_attr')

        self.retrieve(_name_='dropcaslib', caslib=cas_lib_name)

    def model_info(self):

        '''
        Function to return the information of the model table.
        '''

        return self.retrieve(_name_='modelinfo', modelTable=self.model_name)

    def fit(self, data, inputs='_image_', target='_label_',
            mini_batch_size=1, max_epochs=5, log_level=3, lr=0.01,
            optimizer=None,
            **kwargs):

        '''
        Train the deep learning model using the given data.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                a dictionary specifies the CAS table, or an Image object.
            Specifies the training data for the model.
        inputs : string, optional
            Specifies the variable name of in the input_tbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specifies the variable name of in the input_tbl, that is the response of the deep learning model.
            Default : '_label_'.
        mini_batch_size : integer, optional
            Specifies the number of observations per thread in a mini-batch..
            Default : 1.
        max_epochs : int64, optional
            Specifies the maximum number of Epochs.
            Default : 5.
        log_level : int 0-3, optional
            Specifies  how progress messages are sent to the client.
                0 : no messages are sent.
                1 : send the start and end messages.
                2 : send the iteration history for each Epoch.
                3 : send the iteration history for each batch.
            Default : 3.
        lr : double, optional
            Specifies the learning rate of the algorithm.
            Default : 0.01.
        optimizer: dictionary, optional
            Specifies the options for the optimizer in the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_dlcommon_dlOptimizerOpts.html
            for detail.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlTrain.html
            for detail.

        Returns

        ----------
        Return a fetch result to the client, about the trained model summary.
        The updated model weights are automatically assigned to the Model object.

        '''

        input_tbl = input_table_check(data)

        if optimizer is None:
            optimizer = dict(algorithm=dict(learningrate=lr),
                             minibatchsize=mini_batch_size,
                             maxepochs=max_epochs,
                             loglevel=log_level)
        elif isinstance(optimizer, dict):
            optimizer = dict((k.lower(), v) for k, v in optimizer.items())
            opt_keys = optimizer.keys()
            if 'minibatchsize' not in opt_keys:
                optimizer['minibatchsize'] = mini_batch_size
            if 'maxepochs' not in opt_keys:
                optimizer['maxepochs'] = max_epochs
            if 'loglevel' not in opt_keys:
                optimizer['loglevel'] = log_level
            if 'algorithm' in opt_keys:
                algorithm = dict((k.lower(), v) for k, v in optimizer['algorithm'].items())
                alg_keys = algorithm.keys()
                if 'learningrate' not in alg_keys:
                    algorithm['learningrate'] = lr
                optimizer['algorithm'] = algorithm
            else:
                optimizer['algorithm']['learningrate'] = lr
        else:
            raise TypeError('optimizer should be a dictionary of optimization options.')

        max_epochs = optimizer['maxepochs']

        train_options = dict(model=self.model_name,
                             table=input_tbl,
                             inputs=inputs,
                             target=target,
                             modelWeights=dict(replace=True, **self.model_weights.to_table_params()),
                             optimizer=optimizer,
                             **kwargs)

        if self.model_weights.to_table_params()['name'].upper() in \
                list(self.retrieve(_name_='tableinfo').TableInfo.Name):
            print('NOTE: Training based on existing weights.')
            train_options['initWeights'] = self.model_weights
        else:
            print('NOTE: Training from scratch.')

        r = self.retrieve(message_level='note', _name_='dltrain', **train_options)

        temp = r.OptIterHistory
        temp.Epoch += 1  # Epochs should start from 1
        temp.Epoch = temp.Epoch.astype('int64')  # Epochs should be integers

        if self.n_epochs == 0:
            self.n_epochs = max_epochs
            self.training_history = temp
        else:
            temp.Epoch += self.n_epochs
            self.training_history = self.training_history.append(temp)
            self.n_epochs += max_epochs
        self.training_history.index = range(0, self.n_epochs)

        return r

    def tune(self, data, inputs='_image_', target='_label_', **kwargs):

        r = self.retrieve(_name_='dltune',
                          message_level='note', model=self.model_name,
                          table=data,
                          inputs=inputs,
                          target=target,
                          **kwargs)

        return r

    def plot_training_history(self, items=('Loss', 'FitError'), fig_size=(12, 5)):

        '''
        Function to display the training iteration history.
        '''

        self.training_history.plot(x=['Epoch'], y=list(items),
                                   xticks=self.training_history.Epoch,
                                   figsize=fig_size)

    def predict(self, data, inputs='_image_', target='_label_', **kwargs):

        '''
        Function of scoring the deep learning model on a validation data set.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                      a dictionary specifies the CAS table, or an Image object.
            Specifies the validating data for the prediction.
        inputs : string, optional
            Specifies the variable name of in the data, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specifies the variable name of in the data, that is the response of the deep learning model.
            Default : '_label_'.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.


        '''

        input_tbl = input_table_check(data)
        input_tbl = self.conn.CASTable(**input_tbl)
        copy_vars = input_tbl.columns.tolist()

        valid_res_tbl = random_name('Valid_Res')
        dlscore_options = dict(model=self.model_name, initWeights=self.model_weights,
                               table=input_tbl,
                               copyVars=copy_vars,
                               randomFlip='NONE',
                               randomCrop='NONE',
                               casout=dict(replace=True, name=valid_res_tbl),
                               encodeName=True)
        dlscore_options.update(kwargs)

        res = self.retrieve(_name_='dlscore', **dlscore_options)

        self.valid_score = res.ScoreInfo
        self.valid_conf_mat = self.conn.crosstab(
            table=valid_res_tbl, row=target, col='I_' + target)

        temp_tbl = self.conn.CASTable(valid_res_tbl)
        temp_columns = temp_tbl.columninfo().ColumnInfo.Column

        columns = [item for item in temp_columns if item[0:9] == 'P_' + target or item == 'I_' + target]
        img_table = self.retrieve(_name_='fetchimages', fetchimagesvars=columns,
                                  imagetable=temp_tbl, to=1000)
        img_table = img_table.Images

        self.valid_res = img_table

        return res

    def plot_predict_res(self, type='A', image_id=0):
        '''
        Function to plot the classification results.

        Parameters:

        ----------
        type : str, optional.
            Specifies the type of classification results to plot.
            A : All type of results;
            C : Correctly classified results;
            M : Miss classified results.

        image_id : int, optional.
            Specifies the image to be displayed, starting from 0.

        '''
        from .utils import plot_predict_res

        if type == 'A':
            img_table = self.valid_res
        elif type == 'C':
            img_table = self.valid_res
            img_table = img_table[img_table['Label'] == img_table['I__label_']]
        elif type == 'M':
            img_table = self.valid_res
            img_table = img_table[img_table['Label'] != img_table['I__label_']]
        else:
            raise ValueError('type must be one of the following:\n'
                             'A: for all the images\n'
                             'C: for correctly classified images\n'
                             'M: for misclassified images\n')
        img_table.index = range(len(img_table.index))

        columns_for_pred = [item for item in img_table.columns if item[0:9] == 'P__label_']

        image = img_table['Image'][image_id]
        label = img_table['Label'][image_id]

        labels = [item[9:].title() for item in columns_for_pred]
        values = np.asarray(img_table[columns_for_pred].iloc[image_id])
        values, labels = zip(*sorted(zip(values, labels)))
        values = values[-5:]
        labels = labels[-5:]
        labels = [item[:(item.find('__') > 0) * item.find('__') + (item.find('__') < 0) * len(item)]
                  for item in labels]
        labels = [item.replace('_', '\n') for item in labels]

        plot_predict_res(image, label, labels, values)

    def get_feature_maps(self, data, label=None, image_id=0, **kwargs):
        '''
        Function to extract the feature maps for a single image.

        Parameters:

        ----------
        data : An ImageTable object.
            Specifies the table containing the image data.
        label: str, optional
            Specifies the which class of image to use.
            Default : None
        image_id : int, optional
            Specifies which image to use in the table.
            Default : 1.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        Return an instance variable of the Model object, which is a feature map object.
        '''

        uid = data.uid
        if label is None:
            label = uid.iloc[0, 0]
        uid = uid.loc[uid['_label_'] == label]

        if image_id >= uid.shape[0]:
            raise ValueError('image_id should be an integer between 0'
                             ' and {}.'.format(uid.shape[0] - 1))
        uid_value = uid.iloc[image_id, 1]
        uid_name = uid.columns[1]

        input_tbl = input_table_check(data)

        feature_maps_tbl = random_name('Feature_Maps') + '_{}'.format(image_id)
        score_options = dict(model=self.model_name, initWeights=self.model_weights,
                             table=dict(where='{}="{}"'.format(uid_name, uid_value), **input_tbl),
                             layerOut=dict(name=feature_maps_tbl),
                             randomFlip='NONE',
                             randomCrop='NONE',
                             layerImageType='jpg',
                             encodeName=True,
                             **kwargs)
        self.retrieve(_name_='dlscore', **score_options)
        layer_out_jpg = self.conn.CASTable(feature_maps_tbl)
        feature_maps_names = [i for i in layer_out_jpg.columninfo().ColumnInfo.Column]
        feature_maps_structure = dict()
        for feature_map_name in feature_maps_names:
            feature_maps_structure[int(feature_map_name.split('_')[2])] = int(feature_map_name.split('_')[4]) + 1

        self.feature_maps = FeatureMaps(self.conn, feature_maps_tbl, structure=feature_maps_structure)

    def get_features(self, data, dense_layer, target='_label_', **kwargs):
        '''
        Function to extract the features for a data table.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                    a dictionary specifies the CAS table, or an Image object.
            Specifies the table containing the image data.
        dense_layer : str
            Specifies the name of the layer that is extracted.
        target : str, optional
            Specifies the name of the column including the response variable.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        x : ndarray of size n by p, where n is the sample size and p is the number of features.
            The features extracted by the model at the specified dense_layer.
        y : ndarray of size n.
            The response variable of the original data.
        '''

        input_tbl = input_table_check(data)
        feature_tbl = random_name('Features')
        score_options = dict(model=self.model_name, initWeights=self.model_weights,
                             table=dict(**input_tbl),
                             layerOut=dict(name=feature_tbl),
                             layerList=dense_layer,
                             layerImageType='wide',
                             randomFlip='NONE',
                             randomCrop='NONE',
                             encodeName=True,
                             **kwargs)
        self.retrieve(_name_='dlscore', **score_options)
        x = self.conn.CASTable(feature_tbl).as_matrix()
        y = self.conn.CASTable(**input_tbl)[target].as_matrix().ravel()

        return x, y

    def heat_map_analysis(self, data, mask_width=None, mask_height=None, step_size=None):
        '''
        Function to create a heat map on the image, indicating the important region related with classification.


        Parameters:

        ----------
        data : A ImageTable object, containing the column of '_image_', '_label_','_filename_0'
            Specifies the table containing the image data.
        dense_layer : str
            Specifies the name of the layer that is extracted.
        target : str, optional
            Specifies the name of the column including the response variable.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        x : ndarray of size n by p, where n is the sample size and p is the number of features.
            The features extracted by the model at the specified dense_layer.
        y : ndarray of size n.
            The response variable of the original data.
        '''

        output_width = int(data.image_summary.minWidth)
        output_height = int(data.image_summary.minHeight)

        if (data.image_summary.maxWidth != output_width) or \
                (data.image_summary.maxHeight != output_height):
            raise ValueError('Input images must have save sizes.')

        if mask_width is None:
            mask_width = int(output_width / 10)
        if mask_height is None:
            mask_height = int(output_height / 10)
        if step_size is None:
            step_size = int(mask_width / 2)

        copy_vars = data.columns.tolist()
        masked_image_table = random_name('MASKED_IMG')
        blocksize = image_blocksize(output_width, output_height)
        self.retrieve(_name_='image.augmentImages',
                      table=data.to_table_params(),
                      copyvars=copy_vars,
                      casout=dict(replace=True, name=masked_image_table, blocksize=blocksize),
                      cropList=[dict(sweepImage=True, x=0, y=0,
                                     width=mask_width, height=mask_height, stepsize=step_size,
                                     outputwidth=output_width, outputheight=output_height,
                                     mask=True)])

        masked_image_table = self.conn.CASTable(masked_image_table)
        copy_vars = masked_image_table.columns.tolist()
        copy_vars.remove('_image_')
        valid_res_tbl = random_name('Valid_Res')
        dlscore_options = dict(model=self.model_name, initWeights=self.model_weights,
                               table=masked_image_table,
                               copyVars=copy_vars,
                               randomFlip='NONE',
                               randomCrop='NONE',
                               casout=dict(replace=True, name=valid_res_tbl),
                               encodeName=True)
        self.retrieve(_name_='dlscore', **dlscore_options)

        col_list = self.conn.CASTable(valid_res_tbl).columns.tolist()
        temp_table = self.conn.CASTable(valid_res_tbl)[col_list].to_frame()
        image_name_list = temp_table['_filename_0'].unique().tolist()

        prob_tensor = np.empty((output_width, output_height,
                                int(int((output_width - mask_width) / step_size + 1)
                                    * int((output_width - mask_height) / step_size + 1))))
        prob_tensor[:] = np.nan
        model_explain_table = dict()
        count_for_subject = dict()
        for name in image_name_list:
            model_explain_table.update({'{}'.format(name): prob_tensor.copy()})
            count_for_subject.update({'{}'.format(name): 0})

        for row in temp_table.iterrows():
            row = row[1]
            name = row['_filename_0']
            x = int(row['x'])
            y = int(row['y'])
            x_step = int(row['width'])
            y_step = int(row['height'])
            true_pred_prob_col = 'P__label_' + row['_label_']
            prob = row[true_pred_prob_col]
            model_explain_table[name][x:x + x_step, y:y + y_step, count_for_subject[name]] = prob
            count_for_subject[name] += 1

        original_image_table = data.fetchimages(fetchVars=data.columns.tolist()).Images

        output_table = []
        for name in model_explain_table.keys():
            temp_dict = dict()
            temp_dict.update({'_filename_0': name})
            index = original_image_table['_filename_0'] == name
            # print(index)
            temp_dict.update({'_image_': original_image_table['Image'][index].tolist()[0]})
            temp_dict.update({'_label_': original_image_table['Label'][index].tolist()[0]})
            temp_dict.update({'heat_map': np.nanmean(model_explain_table[name], axis=2)})
            output_table.append(temp_dict)

        self.retrieve(_name_='droptable', name=masked_image_table)
        self.retrieve(_name_='droptable', name=valid_res_tbl)

        output_table = pd.DataFrame(output_table)
        self.model_explain_table = output_table

        return output_table

    def plot_heat_map(self, image_id=0, alpha=.2):

        '''
        Function to plot the heat map analysis results.

        Parameters:

        ----------
        image_id : int, optional
            Specifies the image to be displayed, starting from 0.
        alpha : double, between 0 and 1, optional
            Specifies transparent ratio of the overlayed image.

        Returns

        ----------
        A plot of three images, orignal, overlay and heatmap, from left to right.

        '''
        label = self.model_explain_table['_label_'][image_id]

        img = self.model_explain_table['_image_'][image_id]
        img_size = img.size
        extent = [0, img_size[0], 0, img_size[1]]

        heat_map = self.model_explain_table['heat_map'][image_id]
        vmin = heat_map.min()
        fig, (ax0, ax2, ax1) = plt.subplots(ncols=3, figsize=(12, 4))
        ax0.imshow(img, extent=extent)
        ax0.axis('off')
        ax0.set_title('Original Image: {}'.format(label))

        color_bar = ax1.imshow(heat_map, vmax=1, vmin=vmin, interpolation='none', extent=extent)
        ax1.axis('off')
        ax1.set_title('Heat Map')

        ax2.imshow(img, extent=extent)
        ax2.imshow(heat_map, vmax=1, vmin=vmin, interpolation='none', alpha=alpha, extent=extent)
        ax2.axis('off')
        ax2.set_title('Overlayed Image')

        box = ax1.get_position()
        ax3 = fig.add_axes([box.x1 * 1.02, box.y0 + box.height * 0.06, box.width * 0.05, box.height * 0.88])
        plt.colorbar(color_bar, cax=ax3)

        plt.show()

    def save_to_astore(self, path=None):
        '''
        Function to save the model to an astore object, and write it into a file.

         Parameters:

        ----------
        path: str
            Specifies the name of the path to store the model astore.
        '''

        if not self.conn.queryactionset('astore')['astore']:
            self.conn.loadactionset('astore', _messagelevel='error')

        CAS_tbl_name = self.model_name + '_astore'

        self.retrieve(_name_='dlexportmodel',
                      casout=dict(replace=True, name=CAS_tbl_name),
                      initWeights=self.model_weights,
                      modelTable=self.model_name)

        model_astore = self.retrieve(_name_='download',
                                     rstore=CAS_tbl_name)

        file_name = self.model_name + '.astore'
        if path is None:
            path = os.getcwd()

        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        with open(file_name, 'wb') as file:
            file.write(model_astore['blob'])
        print('NOTE: Model astore file saved successfully.')

    def save_to_table(self, path):

        '''
        Function to save the model as sas dataset.

        Parameters:

        ----------
        path: str
            Specifies the name of the path to store the model tables.

        Return:

        ----------
        The specified files in the 'CASUSER' library.

        '''

        cas_lib_name = random_name('CASLIB')
        self.retrieve(_name_='addcaslib',
                      activeonadd=False, datasource=dict(srcType="DNFS"), name=cas_lib_name,
                      path=path)

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = _file_name_ + _extension_
        weight_tbl_file = _file_name_ + '_weights' + _extension_
        attr_tbl_file = _file_name_ + '_weights_attr' + _extension_

        self.retrieve(_name_='table.save',
                      table=self.model_name,
                      name=model_tbl_file,
                      replace=True, caslib=cas_lib_name)
        self.retrieve(_name_='table.save',
                      table=self.model_weights,
                      name=weight_tbl_file,
                      replace=True, caslib=cas_lib_name)
        CAS_tbl_name = random_name('Attr_Tbl')
        self.retrieve(_name_='table.attribute',
                      task='CONVERT', attrtable=CAS_tbl_name,
                      **self.model_weights.to_table_params())
        self.retrieve(_name_='table.save',
                      table=CAS_tbl_name,
                      name=attr_tbl_file,
                      replace=True, caslib=cas_lib_name)

        self.retrieve(_name_='dropcaslib',
                      caslib=cas_lib_name)
        print('NOTE: Model table saved successfully.')

    def deploy(self, path, output_format='ASTORE'):
        '''
        Function to deploy the deep learning model.

        Parameters:

        ----------
        path : string,
            Specifies the name of the path to store the model tables or astore.
        format : string, optional.
            specifies the format of the deployed model.
            Supported format: ASTORE, CASTABLE
            Default: ASTORE


        '''

        if output_format.lower() == 'astore':
            self.save_to_astore(path=path)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path=path)
        else:
            raise TypeError('output_format must be "astore", "castable" or "table"')

    def count_params(self):
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

    def summary(self):
        bar_line = '*' + '=' * 18 + '*' + '=' * 15 + '*' + '=' * 8 + '*' + \
                   '=' * 12 + '*' + '=' * 17 + '*' + '=' * 22 + '*\n'
        h_line = '*' + '-' * 18 + '*' + '-' * 15 + '*' + '-' * 8 + '*' + \
                 '-' * 12 + '*' + '-' * 17 + '*' + '-' * 22 + '*\n'
        title_line = '|{:^18}'.format('Layer (Type)') + \
                     '|{:^15}'.format('Kernel Size') + \
                     '|{:^8}'.format('Stride') + \
                     '|{:^12}'.format('Activation') + \
                     '|{:^17}'.format('Output Size') + \
                     '|{:^22}|\n'.format('Number of Parameters')
        output = bar_line + title_line + h_line
        for layer in self.layers:
            output = output + layer.summary_str
        output = output + bar_line

        output = output + '|Total Number of Parameters: {:<69}|\n'. \
            format(format(self.count_params(), ','))
        output = output + '*' + '=' * 97 + '*'

        self.summary_str = output
        print(output)

    def plot_network(self):
        '''
        Function to plot the model DAG
        '''

        from IPython.display import display
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

        display(model_to_graph(self))


class FeatureMaps:
    '''
    A class for feature maps.
    '''

    def __init__(self, conn, feature_maps_tbl, structure=None):
        self.conn = conn
        self.tbl = feature_maps_tbl
        self.structure = structure

    def display(self, layer_id, filter_id=None):
        '''
        Function to display the feature maps.

        Parameters:

        ----------

        layer_id : int
            Specifies the id of the layer to be displayed

        Return:

        ----------
        Plot of the feature maps.


        '''
        from PIL import Image
        from IPython.display import display

        if filter_id is None:
            n_images = self.structure[layer_id]
            filter_id = list(range(n_images))

        if len(filter_id) > 64:
            filter_id = filter_id[0:64]
            print('NOTE: The maximum number of filters to be displayed is 64.\n'
                  'NOTE: Only the first 64 filters are displayed.')

        n_images = len(filter_id)
        n_col = min(n_images, 8)
        n_row = int(np.ceil(n_images / n_col))

        fig = plt.figure(figsize=(16, 16 // n_col * n_row))
        title = 'Activation Maps for Layer_{}'.format(layer_id)

        if layer_id == 0:
            image = []
            for i in range(3):
                col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, i)
                temp = self.conn.retrieve('fetchimages', _messagelevel='error',
                                          table=self.tbl, image=col_name).Images.Image[0]
                image.append(np.asarray(temp))
            image = np.dstack((image[2], image[1], image[0]))
            image = Image.fromarray(image, 'RGB')
            display(image)
        else:
            for i in range(n_images):
                filter_num = filter_id[i]
                col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, filter_num)
                image = self.conn.retrieve('fetchimages', _messagelevel='error',
                                           table=self.tbl, image=col_name).Images.Image[0]
                image = np.asarray(image)
                fig.add_subplot(n_row, n_col, i + 1)
                plt.imshow(image, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.title('Filter {}'.format(filter_num))
            plt.suptitle(title, fontsize=20)


def get_num_configs(keys, layer_type_prefix, layer_table):
    '''
    Function to extract the numerical options from the model table
    Parameters:

    ----------

    keys : list
        Specifies the list of numerical variables.
    layer_type_prefix : str
        Specifies the prefix of the options in the model table.
    layer_table : table
        Specifies the selection of table containing the information for the layer.

    Return:

    ----------
    dictionary of the options that can pass to layer definition.
    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLNumVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' + key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def get_str_configs(keys, layer_type_prefix, layer_table):
    '''
    Function to extract the str options from the model table
    Parameters:

    ----------

    keys : list
        Specifies the list of str variables.
    layer_type_prefix : str
        Specifies the prefix of the options in the model table.
    layer_table : table
        Specifies the selection of table containing the information for the layer.

    Return:

    ----------
    dictionary of the options that can pass to layer definition.
    '''
    layer_config = dict()
    for key in keys:
        try:
            layer_config[key] = layer_table['_DLChrVal_'][
                layer_table['_DLKey1_'] == layer_type_prefix + '.' + key.lower().replace('_', '')].tolist()[0]
        except IndexError:
            pass
    return layer_config


def extract_input_layer(layer_table):
    num_keys = ['n_channels', 'width', 'height', 'dropout', 'scale']
    input_layer_config = dict()
    input_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    input_layer_config.update(get_num_configs(num_keys, 'inputopts', layer_table))

    input_layer_config['offsets'] = []
    try:
        input_layer_config['offsets'].append(
            int(layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'inputopts.offsets'].tolist()[0]))
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'inputopts.offsets.0'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'inputopts.offsets.1'].tolist()[0])
    except IndexError:
        pass
    try:
        input_layer_config['offsets'].append(
            layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'inputopts.offsets.2'].tolist()[0])
    except IndexError:
        pass
    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] == 'inputopts.crop'].tolist()[0] == 'No cropping':
        input_layer_config['random_crop'] = 'NONE'
    else:
        input_layer_config['random_crop'] = 'UNIQUE'

    if layer_table['_DLChrVal_'][layer_table['_DLKey1_'] == 'inputopts.flip'].tolist()[0] == 'No flipping':
        input_layer_config['random_flip'] = 'NONE'
    # else:
    #     input_layer_config['random_flip']='HV'

    layer = InputLayer(**input_layer_config)
    return layer


def extract_conv_layer(layer_table):
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean', 'initbias', 'dropout', 'truncationFactor',
                'initB', 'truncFact']
    str_keys = ['act', 'init']

    conv_layer_config = dict()
    conv_layer_config.update(get_num_configs(num_keys, 'convopts', layer_table))
    conv_layer_config.update(get_str_configs(str_keys, 'convopts', layer_table))
    conv_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'convopts.no_bias'].any():
        conv_layer_config['includeBias'] = False
    else:
        conv_layer_config['includeBias'] = True

    layer = Conv2d(**conv_layer_config)
    return layer


def extract_pooling_layer(layer_table):
    num_keys = ['width', 'height', 'stride']
    str_keys = ['act', 'poolingtype']

    pool_layer_config = dict()
    pool_layer_config.update(get_num_configs(num_keys, 'poolingopts', layer_table))
    pool_layer_config.update(get_str_configs(str_keys, 'poolingopts', layer_table))

    pool_layer_config['pool'] = pool_layer_config['poolingtype'].lower().split(' ')[0]
    del pool_layer_config['poolingtype']
    pool_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Pooling(**pool_layer_config)
    return layer


def extract_batchnorm_layer(layer_table):
    bn_layer_config = dict()
    bn_layer_config.update(get_str_configs(['act'], 'bnopts', layer_table))
    bn_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = BN(**bn_layer_config)
    return layer


def extract_residual_layer(layer_table):
    res_layer_config = dict()

    res_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    res_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Res(**res_layer_config)
    return layer


def extract_fc_layer(layer_table):
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean', 'initbias', 'dropout', 'truncationFactor', 'initB',
                'truncFact']
    str_keys = ['act', 'init']

    fc_layer_config = dict()
    fc_layer_config.update(get_num_configs(num_keys, 'fcopts', layer_table))
    fc_layer_config.update(get_str_configs(str_keys, 'fcopts', layer_table))
    fc_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'fcopts.no_bias'].any():
        fc_layer_config['includeBias'] = False
    else:
        fc_layer_config['includeBias'] = True

    layer = Dense(**fc_layer_config)
    return layer


def extract_output_layer(layer_table):
    num_keys = ['n', 'width', 'height', 'stride', 'std', 'mean', 'initbias', 'dropout', 'truncationFactor', 'initB',
                'truncFact']
    str_keys = ['act', 'init']

    output_layer_config = dict()
    output_layer_config.update(get_num_configs(num_keys, 'outputopts', layer_table))
    output_layer_config.update(get_str_configs(str_keys, 'outputopts', layer_table))
    output_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'outputopts.no_bias'].any():
        output_layer_config['includeBias'] = False
    else:
        output_layer_config['includeBias'] = True

    layer = OutputLayer(**output_layer_config)
    return layer


def layer_to_node(layer):
    cell1 = r'{}\n({})'.format(layer.name, layer.config['type'])

    keys = ['<Act>Activation:', '<Kernel>Kernel Size:']

    content_dict = dict()

    if layer.kernel_size is not None:
        content_dict['<Kernel>Kernel Size:'] = layer.kernel_size

    if layer.activation is not None:
        if 'act' in layer.config:
            content_dict['<Act>Activation:'] = layer.activation
        if 'pool' in layer.config:
            content_dict['<Act>Pooling:'] = layer.activation

    if layer.type_name is not 'Input':
        title_col = '<Output>Output Size:}|'
        value_col = '{}'.format(layer.output_size) + '}'

        for key in keys:
            if key in content_dict.keys():
                title_col = key + '|' + title_col
                value_col = '{}'.format(content_dict[key]) + '|' + value_col
    else:
        title_col = '<Output>Input Size:}|'
        value_col = '{}'.format(layer.output_size) + '}'

    label = cell1 + '|{' + title_col + '{' + value_col
    label = r'{}'.format(label)

    return dict(name=layer.name, label=label, fillcolor=layer.color_code)


def layer_to_edge(layer):
    gv_params = []
    for item in layer.src_layers:
        gv_params.append(dict(tail_name='{}'.format(item.name),
                              head_name='{}'.format(layer.name),
                              len='0.2'))
    return gv_params


def model_to_graph(model):
    import graphviz as gv
    model_graph = gv.Digraph(name=model.model_name,
                             node_attr=dict(shape='record', style='filled,rounded'))
    # can be added later for adjusting figure size.
    # fixedsize='True', width = '4', height = '1'))

    model_graph.attr(label=r'DAG for {}:'.format(model.model_name),
                     labelloc='top', labeljust='left')
    model_graph.attr(fontsize='20')

    for layer in model.layers:
        if layer.config['type'].lower() == 'input':
            model_graph.node(**layer_to_node(layer))
        else:
            model_graph.node(**layer_to_node(layer))
            for gv_param in layer_to_edge(layer):
                model_graph.edge(**gv_param)

    return model_graph
