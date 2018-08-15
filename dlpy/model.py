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

''' Base Model object for deep learning models '''

import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import warnings

from .layers import InputLayer, Conv2d, Pooling, BN, Res, Concat, Dense, OutputLayer
from .utils import image_blocksize, unify_keys, input_table_check, random_name, check_caslib


class Model(object):
    '''
    Model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default : None
    model_weights : CASTable or string or dict
        Specifies the CASTable containing weights of the deep learning model.
        If not specified, random initial will be used.
        Default : None

    Attributes
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_name : str
        Specifies the name of the model used for prediction
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default : None
    model_weights : CASTable or string or dict
        Specifies the CASTable containing weights of the deep learning model.
        If not specified, random initial will be used.
        Default : None
    layers : list
        List of layers in the model
    valid_res : SASDataFrame
        Results after running Model.predict() on client (limeted to 1000)
    valid_res_tbl = CASTable
        Results after running Model.predict() on server
    feature_maps : model.FeatureMaps
        Used to display outputs of individual layers
    valid_conf_mat : CASResults
        Confusion matrix of results
    valid_score : SASDataFrame
        Shows number of Observations, Misclassification Error and Loss Error
    n_epochs : int
        Number of epochs to train
    training_history : SASDataFram
        After running model.fit shows epoch, LearningRate, Loss, and Fiterror
    model_explain_table : pandas DataFrame
        Used for plotting results
    Returns
    -------
    :class:`Model`

    '''

    def __init__(self, conn, model_table=None, model_weights=None):
        if not conn.queryactionset('deepLearn')['deepLearn']:
            conn.loadactionset(actionSet='deeplearn', _messagelevel='error')

        self.conn = conn

        if model_table is None:
            model_table = dict(name=random_name('Model', 6))

        model_table_opts = input_table_check(model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name=random_name('Model', 6)))

        self.model_name = model_table_opts['name']
        self.model_table = model_table_opts

        if model_weights is None:
            self.model_weights = self.conn.CASTable('{}_weights'.format(self.model_name))
        else:
            self.set_weights(model_weights)

        self.layers = []
        self.valid_res = None
        self.valid_res_tbl = None
        self.feature_maps = None
        self.valid_conf_mat = None
        self.valid_score = None
        self.n_epochs = 0
        self.training_history = None
        self.model_explain_table = None

    @classmethod
    def from_table(cls, input_model_table, display_note=True, output_model_table=None):
        '''
        Create a Model object from CAS table that defines a deep learning model

        Parameters
        ----------
        input_model_table : a CAS table object.
            Specifies the CAS table that defines the deep learning model.
        display_note : bool
            Specifies whether to print the note when generating the model table.
        output_model_table : string, dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default : None

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
            layertype = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                  'layertype'].tolist()[0]
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
        output_model_table : string, dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default : None

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
            Default : None
        output_model_table : string, dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default : None
        Returns
        -------
        :class:`Model`

        '''
        from .model_conversion.sas_caffe_parse import caffe_to_sas

        if output_model_table is None:
            output_model_table = dict(name=random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts.keys():
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
                         include_weights=False, input_weights_file=None):
        '''
        Generate a model object from a Keras model object

        Parameters
        ----------
        conn : CAS
            The CAS connection object.
        keras_model : keras_model object.
            Specifies the keras model to be converted.
        output_model_table : string, dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default : None
        include_weights : boolean, optional
            Specifies whether to load the weights of the keras model.
            Default : True
        input_weights_file : string, optional
            A fully specified client side path to the HDF5 file that stores the keras model weights.
            Only effective when include_weights=True.
            If None is given, the current weights in the keras model will be used.
            Default : None

        Returns
        -------
        :class:`Model`

        '''

        from .model_conversion.sas_keras_parse import keras_to_sas
        if output_model_table is None:
            output_model_table = dict(name=random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts.keys():
            model_table_opts.update(**dict(name=random_name('caffe_model', 6)))

        model_name = model_table_opts['name']

        output_code = keras_to_sas(model=keras_model, model_name=model_name)
        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table=input_model_table)

        if include_weights:
            from .model_conversion.write_keras_model_parm import write_keras_hdf5, write_keras_hdf5_from_file
            temp_HDF5 = os.path.join(os.getcwd(), '{}_weights.kerasmodel.h5'.format(model_name))
            if input_weights_file is None:
                write_keras_hdf5(keras_model, temp_HDF5)
            else:
                write_keras_hdf5_from_file(keras_model, input_weights_file, temp_HDF5)
            print('NOTE: the model weights has been stored in the following file:\n'
                  '{}'.format(temp_HDF5))
        return model

    def _retrieve_(self, _name_, message_level='error', **kwargs):
        return self.conn.retrieve(_name_, _messagelevel=message_level, **kwargs)

    def load(self, path, display_note=True):
        '''
        Load the deep learning model architecture from existing table

        Parameters
        ----------
        path: string
            Specifies the absolute server-side path of the table file.
        display_note : bool
            Specifies whether to print the note when generating the model table.

        '''

        dir_name, file_name = os.path.split(path)

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
            layertype = layer_table['_DLNumVal_'][layer_table['_DLKey1_'] ==
                                                  'layertype'].tolist()[0]
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
        if not flag:
            self._retrieve_('table.dropcaslib', caslib=cas_lib_name)

    def set_weights(self, weight_tbl):
        '''
        Assign weights to the Model object

        Parameters
        ----------
        weight_tbl : CASTable or string or dict
            Specifies the weights CAS table for the model

        '''
        weight_tbl = input_table_check(weight_tbl)
        weight_name = self.model_name + '_weights'

        if weight_tbl['name'].lower() != weight_name.lower():
            self._retrieve_('table.partition',
                            casout=dict(replace=True, name=self.model_name + '_weights'),
                            table=weight_tbl)

        self.model_weights = self.conn.CASTable(name=self.model_name + '_weights')
        print('NOTE: Model weights attached successfully!')

    def load_weights(self, path, **kwargs):
        '''
        Load the weights form a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        Notes
        -----
        Currently support HDF5 and sashdat files.

        '''

        dir_name, file_name = os.path.split(path)
        if file_name.lower().endswith('.sashdat'):
            self.load_weights_from_table(path)
        elif file_name.lower().endswith('caffemodel.h5'):
            self.load_weights_from_caffe(path, **kwargs)
        elif file_name.lower().endswith('kerasmodel.h5'):
            self.load_weights_from_keras(path, **kwargs)
        else:
            warnings.warn('Weights file must be one of the follow types:\n'
                          'sashdat, caffemodel.h5 or kerasmodel.h5.\n'
                          'Weights load failed.', RuntimeWarning)

    def load_weights_from_caffe(self, path, **kwargs):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.

        '''
        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True,
                                          name=self.model_name + '_weights'),
                        formatType='CAFFE', weightFilePath=path, **kwargs)

    def load_weights_from_keras(self, path, **kwargs):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.

        '''
        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True,
                                          name=self.model_name + '_weights'),
                        formatType='KERAS', weightFilePath=path, **kwargs)

    def load_weights_from_table(self, path):
        '''
        Load the weights from a file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        '''
        dir_name, file_name = os.path.split(path)
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

        if not flag:
            self._retrieve_('table.dropcaslib', caslib=cas_lib_name)

    def set_weights_attr(self, attr_tbl, clear=True):
        '''
        Attach the weights attribute to the model weights

        Parameters
        ----------
        attr_tbl : CASTable or string or dict
            Specifies the CAS table that contains the weights attribute table
        clear : boolean, optional
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
        dir_name, file_name = os.path.split(path)
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

    def get_model_info(self):
        '''
        Return the information about the model table

        Returns
        -------
        :class:`CASResults`

        '''
        return self._retrieve_('deeplearn.modelinfo', modelTable=self.model_table)

    def fit(self, data, inputs='_image_', target='_label_',
            mini_batch_size=1, max_epochs=5, log_level=3, lr=0.01,
            optimizer=None, **kwargs):
        '''
        Train the deep learning model using the given data

        Parameters
        ----------
        data : CASTable or string or dict
            Specifies the CAS table containing the training data for the model
        inputs : string, optional
            Specifies the variable name of in the input_tbl, that is the
            input of the deep learning model.
            Default : '_image_'
        target : string, optional
            Specifies the variable name of in the input_tbl, that is the
            response of the deep learning model.
            Default : '_label_'
        mini_batch_size : integer, optional
            Specifies the number of observations per thread in a mini-batch
            Default : 1
        max_epochs : int64, optional
            Specifies the maximum number of Epochs
            Default : 5
        log_level : int, optional
            Specifies how progress messages are sent to the client
            0 - no messages are sent.
            1 - send the start and end messages.
            2 - send the iteration history for each Epoch.
            3 - send the iteration history for each batch.
            Default : 3
        lr : double, optional
            Specifies the learning rate of the algorithm
            Default : 0.01
        optimizer : dictionary, optional
            Specifies the options for the optimizer in the dltrain action.
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dltrain action.

        Returns
        ----------
        :class:`CASResults`

        '''
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)
        if target not in input_table.columninfo().ColumnInfo.Column.tolist():
            raise ValueError('Column name "{}" not found in the data table.'.format(target))

        if inputs not in input_table.columninfo().ColumnInfo.Column.tolist():
            raise ValueError('Column name "{}" not found in the data table.'.format(inputs))

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
                algorithm = dict((k.lower(), v)
                                 for k, v in optimizer['algorithm'].items())
                alg_keys = algorithm.keys()
                if 'learningrate' not in alg_keys:
                    algorithm['learningrate'] = lr
                optimizer['algorithm'] = algorithm
            else:
                optimizer['algorithm']['learningrate'] = lr
        else:
            raise TypeError('optimizer should be a dictionary of optimization options.')

        max_epochs = optimizer['maxepochs']

        train_options = dict(model=self.model_table,
                             table=input_tbl_opts,
                             inputs=inputs,
                             target=target,
                             modelWeights=dict(replace=True,
                                               **self.model_weights.to_table_params()),
                             optimizer=optimizer)
        train_options = unify_keys(train_options)
        try:
            kwargs = unify_keys(kwargs)
            train_options.update(kwargs)
        except:
            pass

        if self.model_weights.to_table_params()['name'].upper() in \
                list(self._retrieve_('table.tableinfo').TableInfo.Name):
            print('NOTE: Training based on existing weights.')
            train_options['initWeights'] = self.model_weights
        else:
            print('NOTE: Training from scratch.')

        r = self._retrieve_('deeplearn.dltrain', message_level='note', **train_options)

        try:
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
        except:
            pass

        return r

    def tune(self, data, inputs='_image_', target='_label_', **kwargs):
        '''
        Tunes hyper parameters for the deep learning model.

        Parameters
        ----------
        data : CASTable or string or dict
            Specifies the CAS table containing the training data for the model
        inputs : string, optional
            Specifies the variable name of in the input_tbl, that is the
            input of the deep learning model.
            Default : '_image_'
        target : string, optional
            Specifies the variable name of in the input_tbl, that is the
            response of the deep learning model.
            Default : '_label_'
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dltune action.

        Returns
        ----------
        :class:`CASResults`

        '''
        r = self._retrieve_('deeplearn.dltune',
                            message_level='note', model=self.model_table,
                            table=data,
                            inputs=inputs,
                            target=target,
                            **kwargs)
        return r

    def plot_training_history(self, items=('Loss', 'FitError'), fig_size=(12, 5)):
        '''
        Display the training iteration history.
        
        
        Parameters
        ----------
        items : tuple
            Specifies the items to be displayed.
            Default : ('Loss', 'FitError')
        fig_size : tuple
            Specifies the size of the figure.
            Default : (12, 5)


        '''
        self.training_history.plot(x='Epoch', y=list(items),
                                   xticks=self.training_history.Epoch,
                                   figsize=fig_size)

    def predict(self, data, inputs='_image_', target='_label_', **kwargs):
        '''
        Evaluate the deep learning model on a specified validation data set

        Parameters
        ----------
        data : ImageTable or string or dict
            Specifies the ImageTable containing the validating data
            for the prediction
        inputs : string, optional
            Specifies the variable name of in the data, that is the input
            of the deep learning model.
            Default : '_image_'
        target : string, optional
            Specifies the variable name of in the data, that is the response
            of the deep learning model.
            Default : '_label_'
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dlScore action.


        Returns
        -------
        :class:`CASResults`

        '''
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)
        if target not in input_table.columninfo().ColumnInfo.Column.tolist():
            raise ValueError('Column name "{}" not found in the data table.'.format(target))

        if inputs not in input_table.columninfo().ColumnInfo.Column.tolist():
            raise ValueError('Column name "{}" not found in the data table.'.format(inputs))

        input_table = self.conn.CASTable(**input_tbl_opts)
        copy_vars = input_table.columns.tolist()

        valid_res_tbl = random_name('Valid_Res')
        dlscore_options = dict(model=self.model_table, initweights=self.model_weights,
                               table=input_table,
                               copyvars=copy_vars,
                               randomflip='none',
                               randomcrop='none',
                               casout=dict(replace=True, name=valid_res_tbl),
                               encodename=True)
        try:
            kwargs = unify_keys(kwargs)
        except:
            pass

        dlscore_options.update(kwargs)

        res = self._retrieve_('deeplearn.dlscore', **dlscore_options)

        self.valid_score = res.ScoreInfo
        self.valid_conf_mat = self.conn.crosstab(
            table=valid_res_tbl, row=target, col='I_' + target)

        temp_tbl = self.conn.CASTable(valid_res_tbl)
        self.valid_res_tbl = temp_tbl

        temp_columns = temp_tbl.columninfo().ColumnInfo.Column

        columns = [item for item in temp_columns
                   if item[0:9] == 'P_' + target or item == 'I_' + target]
        columns.append('_filename_0') # include image names
        columns.append('_id_')
        img_table = self._retrieve_('image.fetchimages', fetchimagesvars=columns,
                                    imagetable=temp_tbl, to=1000)
        img_table = img_table.Images

        self.valid_res = img_table

        return res

    def plot_predict_res(self, type='A', image_id=0):
        '''
        Plot the classification results.

        Parameters
        ----------
        type : str, optional.
            Specifies the type of classification results to plot
            A - All type of results
            C - Correctly classified results
            M - Miss classified results
        image_id : int, optional
            Specifies the image to be displayed, starting from 0

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

        columns_for_pred = [item for item in img_table.columns
                            if item[0:9] == 'P__label_']

        image = img_table['Image'][image_id]
        label = img_table['Label'][image_id]

        labels = [item[9:].title() for item in columns_for_pred]
        values = np.asarray(img_table[columns_for_pred].iloc[image_id])
        values, labels = zip(*sorted(zip(values, labels)))
        values = values[-5:]
        labels = labels[-5:]
        labels = [item[:(item.find('__') > 0) * item.find('__') +
                        (item.find('__') < 0) * len(item)] for item in labels]
        labels = [item.replace('_', '\n') for item in labels]

        plot_predict_res(image, label, labels, values)

    def get_feature_maps(self, data, label=None, image_id=0, **kwargs):
        '''
        Extract the feature maps for a single image

        Parameters
        ----------
        data : ImageTable
            Specifies the table containing the image data.
        label: str, optional
            Specifies the which class of image to use.
            Default : None
        image_id : int, optional
            Specifies which image to use in the table.
            Default : 1
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dlScore action.


        '''
        try:
            uid = data.uid
        except:
            raise TypeError("The input data should be an ImageTable.")
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
        score_options = dict(model=self.model_table, initWeights=self.model_weights,
                             table=dict(where='{}="{}"'.format(uid_name,
                                                               uid_value), **input_tbl),
                             layerOut=dict(name=feature_maps_tbl),
                             randomflip='none',
                             randomcrop='none',
                             layerImageType='jpg',
                             encodeName=True,
                             **kwargs)
        self._retrieve_('deeplearn.dlscore', **score_options)
        layer_out_jpg = self.conn.CASTable(feature_maps_tbl)
        feature_maps_names = [i for i in layer_out_jpg.columninfo().ColumnInfo.Column]
        feature_maps_structure = dict()
        for feature_map_name in feature_maps_names:
            feature_maps_structure[int(feature_map_name.split('_')[2])] = \
                int(feature_map_name.split('_')[4]) + 1

        self.feature_maps = FeatureMaps(self.conn, feature_maps_tbl,
                                        structure=feature_maps_structure)

    def get_features(self, data, dense_layer, target='_label_', **kwargs):
        '''
        Extract linear features for a data table from the layer specified by dense_layer

        Parameters
        ----------
        data : CASTable or string or dict
            Specifies the table containing the image data
        dense_layer : string
            Specifies the name of the layer that is extracted
        target : string, optional
            Specifies the name of the column including the response variable
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dlScore action.

        Returns
        -------
        ( nxp-ndarray, n-ndarray )
            The first ndarray is of size n by p, where n is the sample size
            and p is the number of features.  The features extracted by the
            model at the specified dense_layer.  The second ndarray is of
            size n and contains the response variable of the original data.

        '''

        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)
        if target not in input_table.columninfo().ColumnInfo.Column.tolist():
            raise ValueError('Column name "{}" not found in the data table.'.format(target))

        feature_tbl = random_name('Features')
        score_options = dict(model=self.model_table, initWeights=self.model_weights,
                             table=dict(**input_tbl_opts),
                             layerOut=dict(name=feature_tbl),
                             layerList=dense_layer,
                             layerImageType='wide',
                             randomflip='none',
                             randomcrop='none',
                             encodeName=True,
                             **kwargs)
        self._retrieve_('deeplearn.dlscore', **score_options)
        x = self.conn.CASTable(feature_tbl).as_matrix()
        y = self.conn.CASTable(**input_tbl_opts)[target].as_matrix().ravel()
        return x, y


    def heat_map_analysis(self, data=None, mask_width=None, mask_height=None, step_size=None,
                           display=True, img_type='A', image_id=None, filename=None, inputs="_image_",
                           target="_label_", max_display=5, **kwargs):

        '''
        Conduct a heat map analysis on table of images

        Parameters
        ----------
        data : ImageTable object
            If data is None then the results from model.predict are used.
            data specifies the table containing the image data which must contain
            the columns '_image_', '_label_', '_id_' and '_filename_0'.
        mask_width : int
            Specifies the width of the mask which cover the region of the image.
        mask_height : int
            Specifies the height of the mask which cover the region of the image.
        step_size : int
            Specifies the stepsize of the movement of the the mask.
        display : boolean
            Specifies whether to display the results.
        img_type : string
            Can be 'A' for all images, 'C' for only correctly classified images, or
            'M' for misclassified images.
        image_id: list or int
            A unique image id to get the heatmap. A standard column of ImageTable
        filename: list of strings or string
            The name of a file in '_filename_0' if not unique returns multiple
        inputs: string
            name of column for the input into the model.predict function
        target: string
            name of column for the correct label
        max_display: int
            maximum number of images to display. Heatmap takes a significant amount of time
            to run so a max of 5 is default.
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dlScore action.

        Notes
        -----
        Heat map indicates the important region related with classification.
        Details of the process can be found at: https://arxiv.org/pdf/1311.2901.pdf.

        Returns
        -------
        :class:`pandas.DataFrame`
            Contains Columns: ['I__label_', 'P__label_(for each label)', '_filename_0',
           '_id_', '_image_', '_label_', 'heat_map']

        '''

        # if data is passed we get_predictions if data=None we use model.predict() results
        def get_predictions(data=data, inputs=inputs, target=target, kwargs=kwargs):
            input_tbl_opts = input_table_check(data)
            input_table = self.conn.CASTable(**input_tbl_opts)
            if target not in input_table.columns.tolist():
                raise ValueError('Column name "{}" not found in the data table.'.format(target))

            if inputs not in input_table.columns.tolist():
                raise ValueError('Column name "{}" not found in the data table.'.format(inputs))

            input_table = self.conn.CASTable(**input_tbl_opts)
            copy_vars = input_table.columns.tolist()

            valid_res_tbl_com = random_name('Valid_Res_Complete')
            dlscore_options_com = dict(model=self.model_table, initweights=self.model_weights,
                                       table=input_table,
                                       copyvars=copy_vars,
                                       randomflip='none',
                                       randomcrop='none',
                                       casout=dict(replace=True, name=valid_res_tbl_com),
                                       encodename=True)
            try:
                kwargs = unify_keys(kwargs)
            except:
                pass
            dlscore_options_com.update(kwargs)
            res = self._retrieve_('deeplearn.dlscore', **dlscore_options_com)
            return self.conn.CASTable(valid_res_tbl_com)

        from .images import ImageTable

        run_predict = True
        # check input data, if None try to use model.predict results
        if data is None and self.valid_res_tbl is None:
            raise ValueError('No input data and model.predict() has not been run')
        elif data is None:
            print("Using results from model.predict()")
            data = self.valid_res_tbl
            run_predict = False
        elif data.shape[0] == 0:
            raise ValueError('Input table is empty.')


        im_summary = data._retrieve('image.summarizeimages')['Summary']
        output_width = int(im_summary.minWidth)
        output_height = int(im_summary.minHeight)

        if (int(im_summary.maxWidth) != output_width) or \
                (int(im_summary.maxHeight) != output_height):
            raise ValueError('Input images must have same size.')

        if (mask_width is None) and (mask_height is None):
            mask_width = max(int(output_width / 4), 1)
            mask_height = max(int(output_height / 4), 1)
        if mask_width is None:
            mask_width = mask_height
        if mask_height is None:
            mask_height = mask_width

        if step_size is None:
            step_size = max(int(mask_width / 4), 1)

        # Used on calculating probs for masked images only need imageTable columns
        copy_vars = ImageTable.from_table(data).columns.tolist()

        masked_image_table = random_name('MASKED_IMG')
        blocksize = image_blocksize(output_width, output_height)

        #   if image_id does not exist but filename does, create image_id from filename
        if filename and image_id:
            print(" image_id supersedes filename, image_id being used")
        elif filename:
            temp = data[data['_filename_0'].isin(filename)]
            image_id = temp['_id_'].tolist()
            if not image_id:
                raise ValueError('filename: {} not found in table'.format(filename))


        # filter images by id number
        if image_id:
            data = data[data['_id_'].isin(image_id)]
            if data.numrows().numrows == 0:
                raise ValueError('image_id: {} not found in the table'.format(image_id))


        # filter images by id number
        if image_id:
            data = data[data['_id_'].isin(image_id)]
            if data.numrows().numrows == 0:
                raise ValueError('image_id not found in the table')

        # if data was passed in, run predict, otherwise predict has been run using model.predict()
        if run_predict:
            print("Running prediction ...")
            data = get_predictions(data)
            print("... finished running prediction")

            # filter on M / C on scored data
        table_vars = data.columns.tolist()
        if 'I__label_' in table_vars and img_type == 'C':
            data_temp = data[data['_label_'] == data['I__label_']]
            if data_temp.numrows().numrows != 0:
                data = data_temp
            else:
                raise ValueError('No Correct Labels to Heatmap')

        elif 'I__label_' in table_vars and img_type == 'M':
            data_temp = data[data['_label_'] != data['I__label_']]
            if data_temp.numrows().numrows != 0:
                data = data_temp
            else:
                raise ValueError('No Misclassified Data to Heatmap')

        # get max_display number of random images
        # do not want to use two_way_split because converts to imagetable and lose columns
        if data.numrows().numrows > max_display:
            print('NOTE: The number of images in the table is too large,'
                  ' only {} randomly selected images are used in analysis.'.format(max_display))

            te_rate = max_display / data.numrows().numrows * 100

            if not self.conn.queryactionset('sampling')['sampling']:
                self.conn.loadactionset('sampling', _messagelevel='error')

            sample_tbl = random_name('SAMPLE_TBL')
            self._retrieve_('sampling.srs',
                            table=data.to_table_params(),
                            output=dict(casout=dict(replace=True, name=sample_tbl,
                                blocksize=blocksize), copyvars='all'),
                            samppct=te_rate)
            # from .images import ImageTable
            data= self.conn.CASTable(sample_tbl)

        #     print("data head", data.head())

        # Prepare masked images for analysis.
        self._retrieve_('image.augmentimages',
                        table=data.to_table_params(),
                        copyvars=copy_vars,
                        casout=dict(replace=True, name=masked_image_table,
                                    blocksize=blocksize),
                        cropList=[dict(sweepImage=True, x=0, y=0,
                                       width=mask_width, height=mask_height,
                                       stepsize=step_size,
                                       outputwidth=output_width,
                                       outputheight=output_height,
                                       mask=True)])

        masked_image_table = self.conn.CASTable(masked_image_table)
        copy_vars = masked_image_table.columns.tolist()

        #     print("head masked_tbl ", masked_image_table.head())

        copy_vars.remove('_image_')
        valid_res_tbl = random_name('Valid_Res')
        dlscore_options = dict(model=self.model_table, initWeights=self.model_weights,
                               table=masked_image_table,
                               copyVars=copy_vars,
                               randomflip='none',
                               randomcrop='none',
                               casout=dict(replace=True, name=valid_res_tbl),
                               encodeName=True)
        dlscore_options.update(kwargs)
        self._retrieve_('deeplearn.dlscore', **dlscore_options)

        # valid_res_tbl is on all masks
        valid_res_tbl = self.conn.CASTable(valid_res_tbl)

        # brings to client but now down to a few images and we need to display
        temp_table = valid_res_tbl.to_frame()
        # _parentId_ column is automatically added during dlscore based on _id_ column
        image_id_list = temp_table['_parentId_'].unique().tolist()
        n_masks = len(temp_table['_id_'].unique())

        # setup heatmap image
        prob_tensor = np.empty((output_width, output_height, n_masks))
        prob_tensor[:] = np.nan
        model_explain_table = dict()
        count_for_subject = dict()

        for name in image_id_list:
            model_explain_table.update({'{}'.format(name): prob_tensor.copy()})
            count_for_subject.update({'{}'.format(name): 0})

        for row in temp_table.iterrows():
            row = row[1]
            name = str(row['_parentId_'])
            x = int(row['x'])
            y = int(row['y'])
            x_step = int(row['width'])
            y_step = int(row['height'])
            true_class = row['_label_'].replace(' ', '_')
            true_pred_prob_col = 'P__label_' + true_class
            prob = row[true_pred_prob_col]
            model_explain_table[name][y:min(y + y_step, output_height),
            x:min(x + x_step, output_width),
            count_for_subject[name]] = prob
            count_for_subject[name] += 1

        original_image_table = data.fetchimages(fetchVars=data.columns.tolist(),
                                                to=data.numrows().numrows).Images

        # get columns from validation results that show probabilities of all labels
        prob_cols = []
        for col in data.columns:
            if 'P__label' in col:
                prob_cols.append(col)

        # set values for output_table to return to user
        output_table = []
        for id_num in model_explain_table.keys():
            temp_dict = dict()
            temp_dict.update({'_id_': id_num})
            index = original_image_table['_id_'] == int(id_num)
            temp_dict.update({
                '_filename_0': original_image_table['_filename_0'][index].tolist()[0],
                '_image_': original_image_table['Image'][index].tolist()[0],
                '_label_': original_image_table['Label'][index].tolist()[0],
                'I__label_': original_image_table['I__label_'][index].tolist()[0],
                'heat_map': np.nanmean(model_explain_table[id_num], axis=2)
            })
            index2 = data['_id_'] == id_num
            for col_name in prob_cols:
                temp_dict.update({'{}'.format(col_name): data[col_name][index2].tolist()[0]})

            output_table.append(temp_dict)

        self._retrieve_('table.droptable', name=masked_image_table)
        self._retrieve_('table.droptable', name=valid_res_tbl)

        output_table = pd.DataFrame(output_table)
        self.model_explain_table = output_table

        if display:
            n_images = output_table.shape[0]
            if n_images > max_display:
                print('NOTE: Only the results from the first {} images are displayed.'.format(max_display))
                n_images = max_display
            fig, axs = plt.subplots(ncols=3, nrows=n_images, figsize=(12, 4 * n_images))
            if n_images == 1:
                axs = [axs]
            for im_idx in range(n_images):
                label = output_table['_label_'][im_idx]
                pred_label = output_table['I__label_'][im_idx]
                id_num = output_table['_id_'][im_idx]
                filename = output_table['_filename_0'][im_idx]
                img = output_table['_image_'][im_idx]
                heat_map = output_table['heat_map'][im_idx]
                img_size = heat_map.shape
                extent = [0, img_size[0], 0, img_size[1]]

                vmin = heat_map.min()
                vmax = heat_map.max()

                axs[im_idx][0].imshow(img, extent=extent)
                axs[im_idx][0].axis('off')
                axs[im_idx][0].set_title('Original Image: {}'.format(label))

                color_bar = axs[im_idx][2].imshow(heat_map, vmax=vmax, vmin=vmin,
                                                  interpolation='none',
                                                  extent=extent, cmap='jet_r')
                axs[im_idx][2].axis('off')
                axs[im_idx][2].set_title('Heat Map')

                axs[im_idx][1].imshow(img, extent=extent)
                axs[im_idx][1].imshow(heat_map, vmax=vmax, vmin=vmin,
                                      interpolation='none', alpha=0.5,
                                      extent=extent, cmap='jet_r')
                axs[im_idx][1].axis('off')
                axs[im_idx][1].set_title('Overlayed Image')

                box = axs[im_idx][2].get_position()

                ax3 = fig.add_axes([box.x1 * 1.02, box.y0 + box.height * 0.06,
                                    box.width * 0.05, box.height * 0.88])

                plt.colorbar(color_bar, cax=ax3)

                left, width = .0, 1.0
                bottom, height = -.14, .2
                right = left + width
                top = bottom + height
                p = patches.Rectangle(
                    (left, bottom), width, height,
                    fill=False, transform=axs[im_idx][0].transAxes, clip_on=False
                )

                output_str = 'Predicted Label: {}'.format(pred_label)
                output_str += ', filename: {}'.format(filename)
                output_str += ', image_id: {},'.format(id_num)

                axs[im_idx][0].text(left, 0.5 * (bottom + top), output_str,
                                    horizontalalignment='left',
                                    verticalalignment='center',
                                    fontsize=14, color='black',
                                    transform=axs[im_idx][0].transAxes)

            plt.show()
        return output_table

    def plot_heat_map(self, image_id=0, alpha=.2):
        '''
        Display the heat maps analysis results

        Parameters
        ----------
        image_id : int, optional
            Specifies the image to be displayed, starting from 0.
        alpha : double, between 0 and 1, optional
            Specifies transparent ratio of the heat map in the overlayed image.

        Notes
        ----------
        Displays plot of three images: original, overlayed image and heat map,
        from left to right.

        '''
        label = self.model_explain_table['_label_'][image_id]

        img = self.model_explain_table['_image_'][image_id]

        heat_map = self.model_explain_table['heat_map'][image_id]

        img_size = heat_map.shape
        extent = [0, img_size[0], 0, img_size[1]]

        vmin = heat_map.min()
        vmax = heat_map.max()
        fig, (ax0, ax2, ax1) = plt.subplots(ncols=3, figsize=(12, 4))
        ax0.imshow(img, extent=extent)
        ax0.axis('off')
        ax0.set_title('Original Image: {}'.format(label))

        color_bar = ax1.imshow(heat_map, vmax=vmax, vmin=vmin,
                               interpolation='none', extent=extent, cmap='jet_r')
        ax1.axis('off')
        ax1.set_title('Heat Map')

        ax2.imshow(img, extent=extent)
        ax2.imshow(heat_map, vmax=vmax, vmin=vmin, interpolation='none',
                   alpha=alpha, extent=extent, cmap='jet_r')
        ax2.axis('off')
        ax2.set_title('Overlayed Image')

        box = ax1.get_position()
        ax3 = fig.add_axes([box.x1 * 1.02, box.y0 + box.height * 0.06,
                            box.width * 0.05, box.height * 0.88])
        plt.colorbar(color_bar, cax=ax3)

        plt.show()

    def save_to_astore(self, path=None, **kwargs):
        '''
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path: string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.

        '''
        if not self.conn.queryactionset('astore')['astore']:
            self.conn.loadactionset('astore', _messagelevel='error')

        CAS_tbl_name = self.model_name + '_astore'

        self._retrieve_('deeplearn.dlexportmodel',
                        casout=dict(replace=True, name=CAS_tbl_name),
                        initWeights=self.model_weights,
                        modelTable=self.model_table,
                        randomCrop='none',
                        randomFlip='none',
                        **kwargs)

        model_astore = self._retrieve_('astore.download',
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
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        '''
        dir_name, file_name = os.path.split(path)

        try:
            flag, cas_lib_name = check_caslib(self.conn, dir_name)
        except:
            flag = False
            cas_lib_name = random_name('CASLIB')
            self._retrieve_('table.addcaslib',
                            activeonadd=False, datasource=dict(srcType='DNFS'),
                            name=cas_lib_name, path=path)

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = _file_name_ + _extension_
        weight_tbl_file = _file_name_ + '_weights' + _extension_
        attr_tbl_file = _file_name_ + '_weights_attr' + _extension_

        self._retrieve_('table.save',
                        table=self.model_table,
                        name=model_tbl_file,
                        replace=True, caslib=cas_lib_name)
        self._retrieve_('table.save',
                        table=self.model_weights,
                        name=weight_tbl_file,
                        replace=True, caslib=cas_lib_name)
        CAS_tbl_name = random_name('Attr_Tbl')
        self._retrieve_('table.attribute',
                        task='convert', attrtable=CAS_tbl_name,
                        **self.model_weights.to_table_params())
        self._retrieve_('table.save',
                        table=CAS_tbl_name,
                        name=attr_tbl_file,
                        replace=True, caslib=cas_lib_name)
        if not flag:
            self._retrieve_('table.dropcaslib', caslib=cas_lib_name)
        print('NOTE: Model table saved successfully.')

    def deploy(self, path, output_format='astore', **kwargs):
        '''
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables or astore
        output_format : string, optional
            Specifies the format of the deployed model
            Supported format: astore or castable
            Default : astore

        Notes
        -----
        Currently, this function only supports sashdat and astore formats.

        '''
        if output_format.lower() == 'astore':
            self.save_to_astore(path=path, **kwargs)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path=path)
        else:
            raise ValueError('output_format must be "astore", "castable" or "table"')

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

    def print_summary(self):
        ''' Display a table that summarizes the model architecture '''
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
        Display a graph that summarizes the model architecture.

        Returns
        -------
        :class:`graphviz.dot.Digraph`

        '''

        return model_to_graph(self)


class FeatureMaps(object):
    '''
    Feature Maps object

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    feature_maps_tbl : CAS table.
        Specifies the CAS table to store the feature maps.
    structure : dict
        Specifies the structure of the feature maps.

    Returns
    -------
    :class:`FeatureMaps`

    '''

    def __init__(self, conn, feature_maps_tbl, structure=None):
        self.conn = conn
        self.tbl = feature_maps_tbl
        self.structure = structure

    def display(self, layer_id, filter_id=None):
        '''
        Display the feature maps

        Parameters
        ----------
        layer_id : int
            Specifies the id of the layer to be displayed.
        filter_id : list of int
            Specifies the filters to be displayed.
            Default = None

        '''

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
                temp = self.conn.retrieve('image.fetchimages', _messagelevel='error',
                                          table=self.tbl,
                                          image=col_name).Images.Image[0]
                image.append(np.asarray(temp))
            image = np.dstack((image[2], image[1], image[0]))
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])
        else:
            for i in range(n_images):
                filter_num = filter_id[i]
                col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, filter_num)
                image = self.conn.retrieve('image.fetchimages', _messagelevel='error',
                                           table=self.tbl,
                                           image=col_name).Images.Image[0]
                image = np.asarray(image)
                fig.add_subplot(n_row, n_col, i + 1)
                plt.imshow(image, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.title('Filter {}'.format(filter_num))
            plt.suptitle(title, fontsize=20)


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
                'initbias', 'dropout', 'truncationFactor', 'initB', 'truncFact']
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
                'initbias', 'dropout', 'truncationFactor', 'initB', 'truncFact']
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
                'initbias', 'dropout', 'truncationFactor', 'initB', 'truncFact']
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
        gv_params.append(dict(tail_name='{}'.format(item.name),
                              head_name='{}'.format(layer.name),
                              len='0.2'))
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
