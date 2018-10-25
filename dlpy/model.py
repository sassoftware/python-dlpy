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
import numpy as np
import pandas as pd
import warnings
import collections
import sys
from .layers import InputLayer, Conv2d, Pooling, BN, Res, Concat, Dense, OutputLayer, Keypoints, Detection
from .utils import image_blocksize, unify_keys, input_table_check, random_name, check_caslib, caslibify
from .utils import filter_by_image_id, filter_by_filename
from dlpy.utils import DLPyError, Box


class Model(object):
    '''
    Model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
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
        self.feature_maps = None
        self.valid_conf_mat = None
        self.valid_score = None
        self.n_epochs = 0
        self.training_history = None
        self.model_explain_table = None
        self.best_weights = None
        self.valid_res_tbl = None
        self.model_type = 'CNN'
        self.target = None
        self.model_ever_trained = False


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
            elif layer_type == 8:
                model.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layer_type == 9:
                model.layers.append(extract_residual_layer(layer_table=layer_table))
            elif layer_type == 10:
                model.layers.append(extract_concatenate_layer(layer_table=layer_table))
            elif layer_type == 11:
                model.layers.append(extract_detection_layer(layer_table=layer_table))
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
        keras_model : keras_model object
            Specifies the keras model to be converted.
        output_model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        include_weights : bool, optional
            Specifies whether to load the weights of the keras model.
            Default: True
        input_weights_file : string, optional
            A fully specified client side path to the HDF5 file that stores
            the keras model weights. Only effective when include_weights=True.
            If None is given, the current weights in the keras model will be used.
            Default: None

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
        ''' Call a CAS action '''
        return self.conn.retrieve(_name_, _messagelevel=message_level, **kwargs)

    def change_labels(self, label_file, id_column, label_column):
        '''
        Overrides the labels already in the model

        The label_file should be a csv file that has two columns: 1) id
        column that contains ids starting from 0 and 2) label column that
        contains the labels. This file should also have header columns
        and those should be passed to this function (i.e., id_column and
        label_column)

        Parameters
        ----------
        label_file : string
            Specifies the name of the file that contains the new labels.
        id_column : string
            Specifies the name of the id column in label_file.
        label_column : string
            Specifies the name of the label column in label file.

        '''

        if self.model_weights is not None:
            temp_name = random_name('new_label_table', 6)
            temp_model_name = random_name('new_weights_table', 6)
            labels = pd.read_csv(label_file, skipinitialspace=True, index_col=False)
            self.conn.upload_frame(labels, casout=dict(name=temp_name, replace=True),
                                   importoptions={'vars':[
                                       {'name': id_column, 'type': 'int64'},
                                       {'name': label_column, 'type': 'char', 'length': 20}
                                   ]})
            rt = self._retrieve_('deeplearn.dllabeltarget', initWeights=self.model_weights,
                            modelTable=self.model_table, modelWeights=temp_model_name,
                            labelTable=temp_name)

            if rt.severity == 0:
                self.model_weights = self.conn.CASTable(temp_model_name)
            else:
                for m in rt.messages:
                    print(m)
                raise DLPyError('Seems like something went well while changing the labels')
        else:
            raise DLPyError('We do not have any weights yet')

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
        #dir_name, file_name = os.path.split(path)

        #try:
        #    flag, cas_lib_name = check_caslib(self.conn, dir_name)
        #except:
        #flag = False
        #cas_lib_name = random_name('Caslib', 6)
        #self._retrieve_('table.addcaslib',
        #name=cas_lib_name, path=dir_name,
        #activeOnAdd=False, dataSource=dict(srcType='DNFS'))
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

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
            elif layer_type == 8:
                self.layers.append(extract_batchnorm_layer(layer_table=layer_table))
            elif layer_type == 9:
                self.layers.append(extract_residual_layer(layer_table=layer_table))
            elif layer_type == 10:
                self.layers.append(extract_concatenate_layer(layer_table=layer_table))
            elif layer_type == 11:
                self.layers.append(extract_detection_layer(layer_table=layer_table))

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
        #if not flag:
        #    self._retrieve_('table.dropcaslib', caslib=cas_lib_name)

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

    def load_weights(self, path, labels=False):
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
            self.load_weights_from_caffe(path, labels=labels)
        elif file_name.lower().endswith('kerasmodel.h5'):
            self.load_weights_from_keras(path, labels=labels)
        else:
            warnings.warn('Weights file must be one of the follow types:\n'
                          'sashdat, caffemodel.h5 or kerasmodel.h5.\n'
                          'Weights load failed.', RuntimeWarning)

    def load_weights_from_caffe(self, path, labels=False):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        labels : CASTable, optional
            Specifies the table that contains the imagenet1k labels.

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='CAFFE')
        else:
            self.load_weights_from_file(path=path, format_type='CAFFE')

    def load_weights_from_keras(self, path, labels=False):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='KERAS')
        else:
            self.load_weights_from_file(path=path, format_type='KERAS')

    def load_weights_from_file(self, path, format_type='KERAS'):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file

        '''
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True,
                                          name=self.model_name + '_weights'),
                        formatType=format_type, weightFilePath=file_name,
                        caslib=cas_lib_name)

        self.set_weights(self.model_name + '_weights')

    def load_weights_from_file_with_labels(self, path, format_type='KERAS'):
        '''
        Load the model weights from a HDF5 file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the HDF5 file that
            contains the weight table.
        format_type : KERAS, CAFFE
            Specifies the source framework for the weights file

        '''
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

        from dlpy.utils import get_imagenet_labels_table
        label_table = get_imagenet_labels_table(self.conn)

        self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                        modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                        formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                        labelTable=label_table);

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
        cas_lib_name, file_name = caslibify(self.conn, path, task='load')

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

    def fit(self, data, inputs=None, target=None, data_specs=None, mini_batch_size=1, max_epochs=5, log_level=3,
            lr=0.01, optimizer=None, nominals=None, texts=None, target_sequence=None, sequence=None, text_parms=None,
            valid_table=None, valid_freq=1, gpu=None, attributes=None, weight=None, seed=0, record_seed=0,
            missing='mean', target_missing='mean', repeat_weight_table=False, force_equal_padding=None,
            save_best_weights=False, n_threads=None, target_order='ascending'):
        """
        Fitting a deep learning model.

        Parameters
        ----------

        data : string
            This is the input data. It might be a string that is the
            name of a cas table. Alternatively, this might be a cas table.
        inputs : string or list-of-strings, optional
            Specifies the input variables to use in the analysis.
        target : string or list-of-strings, optional
            Specifies the target sequence variables to use in the analysis.
        data_specs : :class:`DataSpec`, optional
            Specifies the parameters for the multiple input cases.
        mini_batch_size : int, optional
            Specifies the number of observations per thread in a
            mini-batch. You can use this parameter to control the number of
            observations that the action uses on each worker for each thread
            to compute the gradient prior to updating the weights. Larger
            values use more memory. When synchronous SGD is used (the
            default), the total mini-batch size is equal to
            miniBatchSize * number of threads * number of workers. When
            asynchronous SGD is used (by specifying the elasticSyncFreq
            parameter), each worker trains its own local model. In this case,
            the total mini-batch size for each worker is
            miniBatchSize * number of threads.
        max_epochs : int, optional
            specifies the maximum number of epochs. For SGD with a
            single-machine server or a session that uses one worker on a
            distributed server, one epoch is reached when the action passes
            through the data one time. For a session that uses more than one
            worker, one epoch is reached when all the workers exchange the
            weights with the controller one time. The syncFreq parameter
            specifies the number of times each worker passes through the
            data before exchanging weights with the controller. For L-BFGS
            with full batch, each L-BFGS iteration might process more than
            one epoch, and final number of epochs might exceed the maximum
            number of epochs.
        log_level : int, optional
            Specifies how progress messages are sent to the client. The
            default value, 0, indicates that no messages are sent. Specify 1
            to receive start and end messages. Specify 2 to include the
            iteration history.
        lr : double, optional
            Specifies the learning rate.
        optimizer : :class:`Optimizer`, optional
            Specifies the parameters for the optimizer.
        nominals : string or list-of-strings, optional
            Specifies the nominal input variables to use in the analysis.
        texts : string or list-of-strings, optional
            Specifies the character variables to treat as raw text.
            These variables must be specified in the inputs parameter.
        target_sequence : string or list-of-strings, optional
            Specifies the target sequence variables to use in the analysis.
        sequence : :class:`Sequence`, optional
            Specifies the settings for sequence data.
        text_parms : :class:`TextParms`, optional
            Specifies the parameters for the text inputs.
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
        attributes : string or list-of-strings, optional
            Specifies temporary attributes, such as a format, to apply to
            input variables.
        weight : string, optional
            Specifies the variable/column name in the input table containing the
            prior weights for the observation.
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
        missing : string, optional
            Specifies the policy for replacing missing values with imputed values.
            Valid Values: MAX, MIN, MEAN, NONE
            Default: MEAN
        target_missing : string, optional
            Specifies the policy for replacing target missing values with
            imputed values.
            Valid Values: MAX, MIN, MEAN, NONE
            Default: MEAN
        repeat_weight_table : bool, optional
            Replicates the entire weight table on each worker node when saving
            weights.
            Default: False
        force_equal_padding : bool, optional
            For convolution or pooling layers, this setting forces left padding
            to equal right padding, and top padding to equal bottom padding.
            This setting might result in an output image that is
            larger than the input image.
            Default: False
        save_best_weights : bool, optional
            When set to True, it keeps the weights that provide the smallest
            loss error.
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.
        target_order : string, optional
            Specifies the order of the labels. It can follow the natural order
            of the labels or order them in the order they are recieved with
            training data samples.
            Valid Values: 'ascending', 'descending', 'hash'
            Default: 'ascending'

        Returns
        --------
        :class:`CASResults`

        """
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)

        if data_specs is None and inputs is None:
            if '_image_' in input_table.columns.tolist():
                print('NOTE: Either dataspecs or inputs need to be non-None, therefore inputs=_image_ is used')
                inputs = '_image_'
            else:
                raise DLPyError('either dataspecs or inputs need to be non-None')

        if optimizer is None:
            optimizer = Optimizer(algorithm=VanillaSolver(learning_rate=lr),  mini_batch_size=mini_batch_size,
                                  max_epochs=max_epochs, log_level=log_level)
        else:
            if not isinstance(optimizer, Optimizer):
                raise DLPyError('optimizer should be an Optimizer object')

        max_epochs = optimizer['maxepochs']

        if target is None and '_label_' in input_table.columns.tolist():
            target = '_label_'

        if self.model_weights.to_table_params()['name'].upper() in \
                list(self._retrieve_('table.tableinfo').TableInfo.Name):
            print('NOTE: Training based on existing weights.')
            init_weights = self.model_weights
        else:
            print('NOTE: Training from scratch.')
            init_weights = None

        if save_best_weights and self.best_weights is None:
            self.best_weights = random_name('model_best_weights', 6)

        r = self.train(table=input_tbl_opts, inputs=inputs, target=target, data_specs=data_specs,
                       optimizer=optimizer, nominals=nominals, texts=texts, target_sequence=target_sequence,
                       sequence=sequence, text_parms=text_parms, valid_table=valid_table, valid_freq=valid_freq,
                       gpu=gpu, attributes=attributes, weight=weight, seed=seed, record_seed=record_seed,
                       missing=missing, target_missing=target_missing, repeat_weight_table=repeat_weight_table,
                       force_equal_padding=force_equal_padding, init_weights=init_weights, target_order=target_order,
                       best_weights=self.best_weights, model=self.model_table, n_threads=n_threads,
                       model_weights=dict(replace=True, **self.model_weights.to_table_params()))

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

        if r.severity < 2:
            self.target = target

        return r

    def train(self, table, attributes=None, inputs=None, nominals=None, texts=None, valid_table=None, valid_freq=1,
              model=None, init_weights=None, model_weights=None, target=None, target_sequence=None,
              sequence=None, text_parms=None, weight=None, gpu=None, seed=0, record_seed=None, missing='mean',
              optimizer=None, target_missing='mean', best_weights=None, repeat_weight_table=False,
              force_equal_padding=None, data_specs=None, n_threads=None, target_order='ascending'):
        """
        Trains a deep learning model

        table : string or CASTable
            Specifies the input data.
        attributes : string or list-of-strings, optional
            Specifies temporary attributes, such as a format, to apply
            to input variables.
        inputs : string or list-of-strings, optional
            Specifies the input variables to use in the analysis.
        nominals : string or list-of-strings
            Specifies the nominal input variables to use in the analysis.
        texts : string or list-of-strings, optional
            Specifies the character variables to treat as raw text.
            These variables must be specified in the inputs parameter.
        valid_table : string or CASTable, optional
            Specifies the table with the validation data. The validation
            table must have the same columns and data types as the
            training table.
        valid_freq : int, optional
            Specifies the frequency for scoring the validation table.
        model : string or CASTable, optional
            Specifies the in-memory table that is the model.
        init_weights : string or CASTable, optional
            Specifies an in-memory table that contains the model weights.
            These weights are used to initialize the model.
        model_weights : string or CASTable, optional
            Specifies an in-memory table that is used to store the
            model weights.
        target : string or list-of-strings, optional
            Specifies the target sequence variables to use in the analysis.
        target_sequence : string or list-of-strings, optional
            Specifies the target sequence variables to use in the analysis.
        sequence : string or list-of-strings, optional
            Specifies the settings for sequence data.
        text_parms : TextParms, optional
            Specifies the parameters for the text inputs.
        weight : string, optional
            Specifies the variable/column name in the input table
            containing the prior weights for the observation.
        gpu : GPU, optional
            When specified, the action uses graphical processing unit hardware.
            The simplest way to use GPU processing is to specify "gpu=1".
            In this case, the default values of other GPU parameters are used.
            Setting gpu=1 enables all available GPU devices for use. Setting
            gpu=0 disables GPU processing.
        seed : double, optional
            specifies the random number seed for the random number
            generator in SGD. The default value, 0, and negative values
            indicate to use random number streams based on the computer
            clock. Specify a value that is greater than 0 for a reproducible
            random number sequence.
        record_seed : double, optional
            specifies the random number seed for the random record
            selection within a worker. The default value 0 disables random
            record selection. Records are read as they are laid out in memory.
            Negative values indicate to use random number streams based
            on the computer clock.
        missing : string, optional
            Specifies the policy for replacing missing values with imputed values.
            Valid Values: MAX, MIN, MEAN, NONE
            Default: MEAN
        optimizer : Optimizer, optional
            Specifies the parameters for the optimizer.
        target_missing : string, optional
            Specifies the policy for replacing target missing values with
            imputed values.
            Valid Values: MAX, MIN, MEAN, NONE
            Default: MEAN
        best_weights : string or CASTable, optional
            Specifies that the weights with the smallest loss error will be
            saved to a CAS table.
        repeat_weight_table : bool, optional
            Replicates the entire weight table on each worker node when
            saving weights.
            Default: False
        force_equal_padding : bool, optional
            For convolutional or pooling layers, this setting forces left padding
            to equal right padding, and top padding to equal bottom padding.
            This setting might result in an output image that is larger than the
            input image. Default: False
        data_specs : DataSpec, optional
            Specifies the parameters for the multiple input cases.
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then all
            of the cores available in the machine(s) will be used.
        target_order : string, optional
            Specifies the order of the labels. It can follow the natural order
            of the labels or order them in the order of the process.
            Valid Values: 'ascending', 'descending', 'hash'
            Default: 'ascending'

        Returns
        -------
        :class:`CASResults`

        """

        b_w = None
        if best_weights is not None:
            b_w = dict(replace=True, name=best_weights)

        parameters = DLPyDict(table=table, attributes=attributes, inputs=inputs, nominals=nominals, texts=texts,
                              valid_table=valid_table, valid_freq=valid_freq, model=model, init_weights=init_weights,
                              model_weights=model_weights, target=target, target_sequence=target_sequence,
                              sequence=sequence, text_parms=text_parms, weight=weight, gpu=gpu, seed=seed,
                              record_seed=record_seed, missing=missing, optimizer=optimizer,
                              target_missing=target_missing, best_weights=b_w, repeat_weight_table=repeat_weight_table,
                              force_equal_padding=force_equal_padding, data_specs=data_specs, n_threads=n_threads,
                              target_order=target_order)

        rt = self._retrieve_('deeplearn.dltrain', message_level='note', **parameters)

        if rt.severity < 2:
            self.model_ever_trained = True

        return rt

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
        items : tuple, optional
            Specifies the items to be displayed.
            Default : ('Loss', 'FitError')
        fig_size : tuple, optional
            Specifies the size of the figure.
            Default : (12, 5)

        '''
        items_not_in_results = [x for x in items if x not in self.training_history.columns]
        if items_not_in_results:
            raise DLPyError('Columns {} are not in results'.format(items_not_in_results))
        if self.training_history is not None:
            self.training_history.plot(x='Epoch', y=list(items),
                                       xticks=self.training_history.Epoch,
                                       figsize=fig_size)
        else:
            raise DLPyError('model.fit should be run before calling plot_training_history')

    def evaluate(self, data, text_parms=None, layer_out=None, layers=None, gpu=None, buffer_size=None,
                 mini_batch_buf_size=None, top_probs=None, use_best_weights=False):
        """
        Evaluate the deep learning model on a specified validation data set

        After the inference, a confusion matrix is created from the results.
        This method is good for classification tasks.

        Parameters
        ----------
        data : string or CASTable, optional
            Specifies the input data.
        text_parms : TextParms, optional
            Specifies the parameters for the text inputs.
        layer_out : string, optional
            Specifies the settings for an output table that includes
            layer output values. By default, all layers are included.
            You can filter the list with the layers parameter.
        layers : list of strings
            Specifies the names of the layers to include in the
            output layers table.
        gpu : GPU, optional
            When specified, the action uses graphical processing
            unit hardware. The simplest way to use GPU processing is
            to specify "gpu=1". In this case, the default values of
            other GPU parameters are used. Setting gpu=1 enables all
            available GPU devices for use. Setting gpu=0 disables GPU
            processing.
        buffer_size : int, optional
            Specifies the number of observations to score in a single
            batch. Larger values use more memory.
            Default: 10
        mini_batch_buf_size : int, optional
            Specifies the size of a buffer that is used to save input data
            and intermediate calculations. By default, each layer allocates
            an input buffer that is equal to the number of input channels
            multiplied by the input feature map size multiplied by the
            bufferSize value. You can reduce memory usage by specifying a
            value that is smaller than the bufferSize. The only disadvantage
            to specifying a small value is that run time can increase because
            multiple smaller matrices must be multiplied instead of a single
            large matrix multiply.
        top_probs : int, optional
            Specifies to include the predicted probabilities along with
            the corresponding labels in the results. For example, if you
            specify 5, then the top 5 predicted probabilities are shown in
            the results along with the corresponding labels.
        use_best_weights : bool, optional
            When set to True, the weights that provides the smallest loss
            error saved during a previous training is used while scoring
            input data rather than the final weights from the training.
            Default: False

        Returns
        -------
        :class:`CASResults`

        """
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)

        copy_vars = input_table.columns.tolist()

        if self.valid_res_tbl is None:
            valid_res_tbl = random_name('Valid_Res')
        else:
            valid_res_tbl = self.valid_res_tbl.name

        lo = None
        if layer_out is not None:
            from swat import CASTable
            if type(layer_out) is CASTable:
                lo = layer_out
            else:
                lo = dict(replace=True, name=layer_out)

        if use_best_weights and self.best_weights is not None:
            print('NOTE: Using the weights providing the smallest loss error.')
            res = self.score(table=input_table, model=self.model_table, init_weights=self.best_weights,
                             copy_vars=copy_vars, casout=dict(replace=True, name=valid_res_tbl),
                             encode_name=True, text_parms=text_parms, layer_out=lo,
                             layers=layers, gpu=gpu, mini_batch_buf_size=mini_batch_buf_size,
                             top_probs=top_probs, buffer_size=buffer_size)
        else:
            if self.model_weights is None:
                raise DLPyError('We need some weights to do scoring.')
            else:
                res = self.score(table=input_table, model=self.model_table, init_weights=self.model_weights,
                                 copy_vars=copy_vars, casout=dict(replace=True, name=valid_res_tbl),
                                 encode_name=True, text_parms=text_parms, layer_out=lo,
                                 layers=layers, gpu=gpu, mini_batch_buf_size=mini_batch_buf_size,
                                 buffer_size=buffer_size, top_probs=top_probs)

        if res.severity > 1:
            raise DLPyError('something is wrong while scoring the input data with the model.')

        if res.ScoreInfo is not None:
            self.valid_score = res.ScoreInfo

        # TODO work on here to make it more user friendly and remove assumptions

        if self.target is not None:
            self.valid_conf_mat = self.conn.crosstab(table=valid_res_tbl, row=self.target, col='I_' + self.target)
        else:
            v = self.conn.CASTable(valid_res_tbl)
            temp_columns = v.columns.tolist()
            output_names = [name for name in temp_columns if (name.startswith('I_'))]
            if len(output_names) > 0:
                self.target = output_names[0][2:]
                self.valid_conf_mat = self.conn.crosstab(table=valid_res_tbl, row=self.target, col='I_' + self.target)

        if self.model_type == 'CNN':
            if not self.conn.has_actionset('image'):
                self.conn.loadactionset(actionSet='image', _messagelevel='error')

            self.valid_res_tbl = self.conn.CASTable(valid_res_tbl)
            temp_columns = self.valid_res_tbl.columns.tolist()
            columns = [item for item in temp_columns if item[0:9] == 'P_' + self.target or item == 'I_' + self.target]
            img_table = self._retrieve_('image.fetchimages', fetchimagesvars=columns, imagetable=self.valid_res_tbl, to=1000)
            img_table = img_table.Images

            self.valid_res = img_table
        else:
            self.valid_res = res

        return res

    def evaluate_object_detection(self, ground_truth, coord_type, detection_data=None, classes=None,
                                  iou_thresholds=np.linspace(0.5, 0.95, 10, endpoint=True)):
        """
        Evaluate the deep learning model on a specified validation data set.

        Parameters
        ----------
        ground_truth : string or CASTable, optional
            Specifies a ground truth table to evaluate its corresponding
            prediction results
        coord_type : string, optional
            Specifies the format of how ground_truth table to represent
            bounding boxes.
            Valid Values: 'yolo', 'coco'
        detection_data : string or CASTable, optional
            Perform evaluation on the table. If the parameter is not specified,
            the function evaluates the last prediction performed
            by the model.
        classes : string or list-of-strings, optional
            The classes are selected to be evaluated. If you never set it,
            then it will perform on all of classes in ground truth table
            and detection_data table.
        iou_thresholds : float or list-of-floats, optional
            Specifying an iou threshold or a list of iou thresholds that
            determines what is counted as a model predicted positive
            detection of the classes defined by classes parameter.

        Returns
        -------
        list containing calculated results.

        """
        if coord_type.lower() not in ['yolo', 'coco']:
            raise ValueError('coord_type, {}, is not supported'.format(coord_type))

        self.conn.update(table=dict(name = self.model_name, where='_DLChrVal_ eq "iouThreshold"'),
                         set=[{'var':'_DLNumVal_', 'value':'0.5'}])

        if detection_data is not None:
            input_tbl_opts = input_table_check(detection_data)
            det_tbl = self.conn.CASTable(**input_tbl_opts)
        elif self.valid_res_tbl is not None:
            det_tbl = self.valid_res_tbl
        else:
            raise DLPyError('Specify detection_data option or do predict() before processing the function')
        det_bb_list = []
        if '_image_' in det_tbl.columns.tolist():
            det_tbl.drop(['_image_'], axis=1, inplace=1)
        for idx, row in det_tbl.iterrows():
            if coord_type.lower() == 'yolo':
                [det_bb_list.append(Box(row.loc['_Object{}_x'.format(i)],
                                        row.loc['_Object{}_y'.format(i)],
                                        row.loc['_Object{}_width'.format(i)],
                                        row.loc['_Object{}_height'.format(i)],
                                        row.loc['_Object{}_'.format(i)],
                                        row.loc['_P_Object{}_'.format(i)],
                                        row.loc['idjoin'])) for i in range(int(row.loc['_nObjects_']))]
            elif coord_type.lower() == 'coco':
                [det_bb_list.append(Box(row.loc['_Object{}_xmin'.format(i)],
                                        row.loc['_Object{}_ymin'.format(i)],
                                        row.loc['_Object{}_xmax'.format(i)],
                                        row.loc['_Object{}_ymax'.format(i)],
                                        row.loc['_Object{}_'.format(i)],
                                        row.loc['_P_Object{}_'.format(i)],
                                        row.loc['idjoin'], 'xyxy')) for i in range(int(row.loc['_nObjects_']))]

        input_tbl_opts = input_table_check(ground_truth)
        gt_tbl = self.conn.CASTable(**input_tbl_opts)
        gt_bb_list = []
        if '_image_' in gt_tbl.columns.tolist():
            gt_tbl.drop(['_image_'], axis=1, inplace=1)
        for idx, row in gt_tbl.iterrows():
            if coord_type.lower() == 'yolo':
                [gt_bb_list.append(Box(row.loc['_Object{}_x'.format(i)],
                                       row.loc['_Object{}_y'.format(i)],
                                       row.loc['_Object{}_width'.format(i)],
                                       row.loc['_Object{}_height'.format(i)],
                                       row.loc['_Object{}_'.format(i)],
                                       1.0,
                                       row.loc['idjoin'])) for i in range(int(row.loc['_nObjects_']))]
            elif coord_type.lower() == 'coco':
                [gt_bb_list.append(Box(row.loc['_Object{}_xmin'.format(i)],
                                       row.loc['_Object{}_ymin'.format(i)],
                                       row.loc['_Object{}_xmax'.format(i)],
                                       row.loc['_Object{}_ymax'.format(i)],
                                       row.loc['_Object{}_'.format(i)],
                                       1.0,
                                       row.loc['idjoin'], 'xyxy')) for i in range(int(row.loc['_nObjects_']))]

        freq_variable = []
        for i in range(int(gt_tbl.max(axis = 1, numeric_only = True)['_nObjects_'])):
            freq_variable.append('_Object{}_'.format(i))
        classes_gt = set(self.conn.freq(gt_tbl, inputs=freq_variable).Frequency['FmtVar'])
        classes_gt = sorted(classes_gt)
        classes_gt = [x for x in classes_gt if not (x is '' or x.startswith('NoObject'))]

        freq_variable = []
        for i in range(int(det_tbl.max(axis=1, numeric_only=True)['_nObjects_'])):
            freq_variable.append('_Object{}_'.format(i))
        use_all_class = False
        if classes is None:
            use_all_class = True
            classes = set(self.conn.freq(det_tbl, inputs=freq_variable).Frequency['FmtVar'])
            classes = sorted(classes)
            classes = [x for x in classes if not (x is '' or x.startswith('NoObject'))]
        elif isinstance(classes, str):
            classes = [classes]
        nrof_classes = len(classes)

        classes_not_detected = [x for x in classes_gt if x not in classes]
        if len([x for x in classes if x not in classes_gt]) > 0:
            raise DLPyError('Detection data contains classes that are not in ground truth')

        if not isinstance(iou_thresholds, collections.Iterable):
            iou_thresholds = [iou_thresholds]
        results = []
        for iou_threshold in iou_thresholds:
            results_iou = []
            for i, cls in enumerate(classes):
                det_bb_cls_list = []
                [det_bb_cls_list.append(bb) for bb in det_bb_list if bb.class_type == cls]  # all of detections of the class
                gt_bb_cls_list = []
                [gt_bb_cls_list.append(bb) for bb in gt_bb_list if bb.class_type == cls]
                det_bb_cls_list = sorted(det_bb_cls_list, key=lambda bb: bb.confidence, reverse=True)
                tp = np.zeros(len(det_bb_cls_list))  # the detections of the class
                fp = np.zeros(len(det_bb_cls_list))
                gt_image_index_list = collections.Counter([bb.image_name for bb in gt_bb_cls_list])
                for key, val in gt_image_index_list.items():
                    gt_image_index_list[key] = np.zeros(val)
                print("Evaluating class: %s (%d detections)" % (str(cls), len(det_bb_cls_list)))
                for idx, det_bb in enumerate(det_bb_cls_list):
                    gt_cls_image_list = [bb for bb in gt_bb_cls_list if bb.image_name == det_bb.image_name]
                    iou_max = sys.float_info.min
                    for j, gt_bb in enumerate(gt_cls_image_list):
                        if Box.iou(det_bb, gt_bb) > iou_max:
                            match_idx = j
                            iou_max = Box.iou(det_bb, gt_bb)
                    if iou_max >= iou_threshold:
                        if gt_image_index_list[det_bb.image_name][match_idx] == 0:
                            tp[idx] = 1
                        gt_image_index_list[det_bb.image_name][match_idx] = 1
                    else:
                        fp[idx] = 1
                acc_tp = np.cumsum(tp)
                acc_fp = np.cumsum(fp)
                precision = np.divide(acc_tp, (acc_tp + acc_fp))
                recall = np.divide(acc_tp, len(gt_bb_cls_list))

                interpolated_precision = [0]
                [interpolated_precision.append(i) for i in precision]
                interpolated_precision.append(0)
                for i in range(len(interpolated_precision) - 1, 0, -1):
                    interpolated_precision[i - 1] = max(interpolated_precision[i - 1], interpolated_precision[i])
                interpolated_precision = interpolated_precision[1:-1]

                recall_level = [i / 10.0 for i in range(10)]
                interpolated_ap = np.interp([i for i in recall_level if i < recall[-1]], recall, interpolated_precision)
                ap_cls = np.sum(interpolated_ap) / 11
                results_class = {
                    'class': cls,
                    'precision': precision,
                    'recall': recall,
                    'AP': ap_cls,
                    'interpolated precision': interpolated_ap,
                    'interpolated recall': recall_level,
                    'total positives': len(gt_bb_cls_list),
                    'total TP': np.sum(tp),
                    'total FP': np.sum(fp)
                }
                results_iou.append(results_class)
            ap_sum = 0
            for i in results_iou:
                ap_sum += i['AP']
            if use_all_class:
                mean_ap = ap_sum / (nrof_classes + len(classes_not_detected))
            else:
                mean_ap = ap_sum / nrof_classes
            results.append({'IoU Threshold': iou_threshold, 'Class Evaluation': results_iou, 'AP': mean_ap})

        return results

    def predict(self, data, text_parms=None, layer_out=None, layers=None, gpu=None, buffer_size=10,
                mini_batch_buf_size=None, top_probs=None, use_best_weights=False, n_threads=None):
        """
        Evaluate the deep learning model on a specified validation data set

        Unlike the `evaluate` function, this function just does the
        inference and does not do further analysis. This function is
        good gor non-classification tasks.

        Parameters
        ----------
        data : string or CASTable, optional
            Specifies the input data.
        text_parms : :class:`TextParms`, optional
            Specifies the parameters for the text inputs.
        layer_out : string, optional
            Specifies the settings for an output table that includes
            layer output values. By default, all layers are included.
            You can filter the list with the layers parameter.
        layers : list of strings
            Specifies the names of the layers to include in the output
            layers table.
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing
            unit hardware. The simplest way to use GPU processing is
            to specify "gpu=1". In this case, the default values of
            other GPU parameters are used. Setting gpu=1 enables all
            available GPU devices for use. Setting gpu=0 disables GPU
            processing.
        buffer_size : int, optional
            Specifies the number of observations to score in a single
            batch. Larger values use more memory.
            Default: 10
        mini_batch_buf_size : int, optional
            Specifies the size of a buffer that is used to save input
            data and intermediate calculations. By default, each layer
            allocates an input buffer that is equal to the number of
            input channels multiplied by the input feature map size
            multiplied by the bufferSize value. You can reduce memory
            usage by specifying a value that is smaller than the
            bufferSize. The only disadvantage to specifying a small
            value is that run time can increase because multiple smaller
            matrices must be multiplied instead of a single large
            matrix multiply.
        top_probs : int, optional
            Specifies to include the predicted probabilities along with
            the corresponding labels in the results. For example, if you
            specify 5, then the top 5 predicted probabilities are shown
            in the results along with the corresponding labels.
        use_best_weights : bool, optional
            When set to True, the weights that provides the smallest loss
            error saved during a previous training is used while scoring
            input data rather than the final weights from the training.
            default: False
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.

        Returns
        -------
        :class:`CASResults`

        """
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)
        copy_vars = input_table.columns.tolist()
        copy_vars = [x for x in copy_vars if not (x.startswith('_Object') or x.startswith('_nObject'))]

        if self.valid_res_tbl is None:
            valid_res_tbl = random_name('Valid_Res')
        else:
            valid_res_tbl = self.valid_res_tbl.name

        lo = None
        if layer_out is not None:
            lo = dict(replace=True, name=layer_out)

        if use_best_weights and self.best_weights is not None:
            print('NOTE: Using the weights providing the smallest loss error.')
            res = self.score(table=input_table, model=self.model_table, init_weights=self.best_weights,
                             copy_vars=copy_vars, casout=dict(replace=True, name=valid_res_tbl), encode_name=True,
                             text_parms=text_parms, layer_out=lo, layers=layers, gpu=gpu,
                             mini_batch_buf_size=mini_batch_buf_size, top_probs=top_probs, buffer_size=buffer_size,
                             n_threads=n_threads)
            self.valid_res_tbl = self.conn.CASTable(valid_res_tbl)
            return res
        else:
            res = self.score(table=input_table, model=self.model_table, init_weights=self.model_weights,
                             copy_vars=copy_vars, casout=dict(replace=True, name=valid_res_tbl), encode_name=True,
                             text_parms=text_parms, layer_out=lo, layers=layers, gpu=gpu,
                             mini_batch_buf_size=mini_batch_buf_size, top_probs=top_probs, buffer_size=buffer_size,
                             n_threads=n_threads)
            self.valid_res_tbl = self.conn.CASTable(valid_res_tbl)
            return res

    def score(self, table, model=None, init_weights=None, text_parms=None, layer_out=None,
              layer_image_type='jpg', layers=None, copy_vars=None, casout=None, gpu=None, buffer_size=10,
              mini_batch_buf_size=None, encode_name=False, random_flip='none', random_crop='none', top_probs=None,
              random_mutation='none', n_threads=None, has_output_term_ids=False, init_output_embeddings=None):
        """
        Inference of input data with the trained deep learning model

        Parameters
        ----------
        table : string or CASTable
            Specifies the input data.
        model : string or CASTable, optional
            Specifies the in-memory table that is the model.
        init_weights : string or CASTable, optional
            Specifies an in-memory table that contains the model weights.
        text_parms : TextParms, optional
            Specifies the parameters for the text inputs.
        layer_out : string, optional
            Specifies the settings for an output table that includes layer
            output values. By default, all layers are included. You can
            filter the list with the layers parameter.
        layer_image_type : string, optional
            Specifies the image type to store in the output layers table.
            JPG means a compressed image (e.g, jpg, png, and tiff)
            WIDE means a pixel per column
            Default: jpg
            Valid Values: JPG, WIDE
        layers : list-of-strings, optional
            Specifies the names of the layers to include in the output
            layers table.
        copy_vars : list-of-strings, optional
            Specifies the variables to transfer from the input table to
            the output table.
        casout :, optional
            Specifies the name of the output table.
        gpu : GPU, optional
            When specified, the action uses graphical processing unit hardware.
            The simplest way to use GPU processing is to specify "gpu=1".
            In this case, the default values of other GPU parameters are used.
            Setting gpu=1 enables all available GPU devices for use. Setting
            gpu=0 disables GPU processing.
        buffer_size : int, optional
            Specifies the number of observations to score in a single
            batch. Larger values use more memory.
            Default: 10
        mini_batch_buf_size : int, optional
            Specifies the size of a buffer that is used to save input data
            and intermediate calculations. By default, each layer allocates
            an input buffer that is equal to the number of input channels
            multiplied by the input feature map size multiplied by the
            bufferSize value. You can reduce memory usage by specifying a
            value that is smaller than the bufferSize. The only disadvantage
            to specifying a small value is that run time can increase because
            multiple smaller matrices must be multiplied instead of a single
            large matrix multiply.
        encode_name : bool, optional
            Specifies whether encoding the variable names in the generated
            casout table such as the predicted probabilities of each
            response variable level.
            Default: False
        random_flip : string, optional
            Specifies how to flip the data in the input layer when image data is used.
            H stands for horizontal
            V stands for vertical
            HW stands for horizontal and vertical
            Approximately half of the input data is subject to flipping.
            Default: NONE
            Valid Values: NONE, H, V, HV
        random_crop : string, optional
            Specifies how to crop the data in the input layer when image
            data is used. Images are cropped to the values that are specified
            in the width and height parameters. Only the images with one or
            both dimensions that are larger than those sizes are cropped.
            UNIQUE: specifies to crop images to the size specified in the
            height and width parameters. Images that are less than or equal
            to the size are not modified. For images that are larger, the
            cropping begins at a random offset for x and y.
            Default: NONE
            Valid Values: NONE, UNIQUE
        top_probs : int, optional
            Specifies to include the predicted probabilities along with
            the corresponding labels in the results. For example, if you
            specify 5, then the top 5 predicted probabilities are shown in
            the results along with the corresponding labels.
        random_mutation : string, optional
            Specifies how to mutate images.
            Default: NONE
            Valid Values: NONE, RANDOM
        n_threads : int, optional
            Specifies the number of threads to use. If nothing is set then
            all of the cores available in the machine(s) will be used.

        Returns
        -------
        :class:`CASResults`

        """

        if self.model_type == 'CNN':
            parameters = DLPyDict(table=table, model=model, init_weights=init_weights, text_parms=text_parms,
                                  layer_image_type=layer_image_type, layers=layers, copy_vars=copy_vars, casout=casout,
                                  gpu=gpu, mini_batch_buf_size=mini_batch_buf_size, buffer_size=buffer_size,
                                  layer_out=layer_out, encode_name=encode_name, n_threads=n_threads, random_flip=random_flip,
                                  random_crop=random_crop, top_probs=top_probs, random_mutation=random_mutation)
        else:
            parameters = DLPyDict(table=table, model=model, init_weights=init_weights, text_parms=text_parms,
                                  layers=layers, copy_vars=copy_vars, casout=casout,
                                  gpu=gpu, mini_batch_buf_size=mini_batch_buf_size, buffer_size=buffer_size,
                                  layer_out=layer_out, encode_name=encode_name, n_threads=n_threads, random_flip=random_flip,
                                  random_crop=random_crop, top_probs=top_probs, random_mutation=random_mutation)

        return self._retrieve_('deeplearn.dlscore', message_level='note', **parameters)

    def plot_evaluate_res(self, cas_table=None, img_type='A', image_id=None, filename=None, n_images=5,
                          target='_label_', predicted_class=None, label_class=None, randomize=False,
                          seed=-1):
        '''
        Plot the bar chart of the classification predictions

        Parameters
        ----------
        cas_table : CASTable, optional
            If None results from model.evaluate are used
            Can pass in another table that has the same
            prediction column names as in model.valid_res_tbl
        img_type : str, optional
            Specifies the type of classification results to plot
            * A - All type of results
            * C - Correctly classified results
            * M - Miss classified results
        image_id : list or int, optional
            Specifies the image by '_id_' column to be displayed
        filename : list of strings or string, optional
            The name of a file in '_filename_0' or '_path_' if not unique
            returns multiple
        n_images : int, optional
            Number of images to evaluate
        target : string, optional
            name of column for the correct label
        predicted_class : string, optional
            Name of desired prediction class to plot results
        label_class : string, optional
            Actual target label of desired class to plot results
        randomize : bool, optional
            If true randomize results
        seed : int, optional
            Random seed used if randomize is true

        '''
        from .utils import plot_predict_res
        # create copy of cas_table so can be dropped after filtering
        if not cas_table:
            if self.valid_res_tbl:
                cas_table = self.valid_res_tbl.partition(casout=dict(name='temp_plot', replace=True))['casTable']
            else:
                raise DLPyError("Need to run model.evaluate()")
        else:
            cas_table = cas_table.partition(casout=dict(name='temp_plot', replace=True))['casTable']

        if target not in cas_table.columns:
            if 'Label' in cas_table.columns:
                target = 'Label'
            else:
                raise DLPyError("target column {} not found in cas_table {}".format(target, cas_table.name))
        if 'I__label_' not in cas_table.columns:
            raise DLPyError("cas_table must contain prediction column named 'I__lable_'."
                            "i.e. model.valid_res_tbl can be used after running model.evaluate")

        filtered = None
        if filename or image_id:

            if '_id_' not in cas_table.columns.tolist():
                print("'_id_' column not in cas_table, processing complete table")
            else:
                if filename and image_id:
                    print(" image_id supersedes filename, image_id being used")
                if image_id:
                    filtered = filter_by_image_id(cas_table, image_id)
                elif filename:
                    filtered = filter_by_filename(cas_table, filename)

            if filtered:
                if filtered.numrows == 0:
                    raise DLPyError(" image_id or filename not found in CASTable {}".format(cas_table.name))
                self.conn.droptable(cas_table)
                cas_table = filtered

        if img_type == 'A':
            if cas_table.numrows().numrows == 0:
                raise DLPyError("No images to heatmap")
        elif img_type == 'C':
            cas_table = cas_table[cas_table[target] == cas_table['I__label_']]
            cas_table = cas_table.partition(casout=dict(name=cas_table.name, replace=True))['casTable']
            if cas_table.numrows().numrows == 0:
                raise DLPyError("No correct labels to heatmap")
        elif img_type == 'M':
            cas_table = cas_table[cas_table[target] != cas_table['I__label_']]
            cas_table.partition(casout=dict(name=cas_table.name, replace=True))['casTable']
            if cas_table.numrows().numrows == 0:
                raise DLPyError("No misclassified labels to heatmap")
        else:
            raise DLPyError('img_type must be one of the following:\n'
                            'A: for all the images\n'
                            'C: for correctly classified images\n'
                            'M: for misclassified images\n')


        if label_class:
            unique_labels = list(set(cas_table[target].tolist()))
            cas_table = cas_table[cas_table['_label_'] == label_class]
            cas_table.partition(casout=dict(name=cas_table.name, replace=True))['casTable']
            if cas_table.numrows().numrows == 0:
                raise DLPyError("There are no labels of {}. The labels consist of {}". \
                                format(label_class, unique_labels))
        if predicted_class:
            unique_predictions = list(set(cas_table['I__label_'].tolist()))
            cas_table = cas_table[cas_table['I__label_'] == predicted_class]
            cas_table.partition(casout=dict(name=cas_table.name, replace=True))['casTable']
            if cas_table.numrows().numrows == 0:
                raise DLPyError("There are no predicted labels of {}. The predicted labels consist of {}". \
                                format(predicted_class, unique_predictions))

        columns_for_pred = [item for item in cas_table.columns
                            if item[0:9] == 'P__label_']
        if len(columns_for_pred) == 0:
            raise DLPyError("Input table has no columns for predictions. "
                            "Run model.predict the predictions are stored "
                            "in the attribute model.valid_res_tbl.")
        fetch_cols = columns_for_pred + ['_id_']

        if randomize:
            cas_table.append_computedvars(['random_index'])
            cas_table.append_computedvarsprogram('call streaminit({});' 'random_index=''rand("UNIFORM")'.format(seed))

            img_table = cas_table.retrieve('image.fetchimages', _messagelevel='error',
                                           table=dict(**cas_table.to_table_params()),
                                           fetchVars=fetch_cols,
                                           sortby='random_index', to=n_images)
        else:
            img_table = cas_table.retrieve('image.fetchimages', fetchVars=fetch_cols, to=n_images,
                                           sortBy=[{'name': '_id_', 'order': 'ASCENDING'}])
        self.conn.droptable(cas_table)
        img_table = img_table['Images']
        for im_idx in range(len(img_table)):
            image = img_table['Image'][im_idx]
            label = 'Correct Label for image {} : {}'.format(img_table['_id_'][im_idx], img_table['Label'][im_idx])
            labels = [item[9:].title() for item in columns_for_pred]
            values = np.asarray(img_table[columns_for_pred].iloc[im_idx])
            values, labels = zip(*sorted(zip(values, labels)))
            values = values[-5:]
            labels = labels[-5:]
            labels = [item[:(item.find('__') > 0) * item.find('__') +
                            (item.find('__') < 0) * len(item)] for item in labels]
            labels = [item.replace('_', '\n') for item in labels]

            plot_predict_res(image, label, labels, values)

    def get_feature_maps(self, data, label=None, idx=0, image_id=None,  **kwargs):
        """
        Extract the feature maps for a single image

        Parameters
        ----------
        data : ImageTable
            Specifies the table containing the image data.
        label : str, optional
            Specifies the which class of image to use.
            Default : None
        idx : int, optional
            Specifies which row index to get feature map
            Default : 1
        image_id : list or int, optional
            Filters data using '_id_' column
        **kwargs : keyword arguments, optional
            Specifies the optional arguments for the dlScore action.

        """
        from .images import ImageTable
        if image_id:
            filtered = filter_by_image_id(data, image_id)
            data = ImageTable.from_table(filtered)
            self.conn.droptable(filtered)

        try:
            uid = data.uid
        except:
            raise TypeError("The input data should be an ImageTable.")
        if label is None:
            label = uid.iloc[0, 0]
        uid = uid.loc[uid['_label_'] == label]

        if idx >= uid.shape[0]:
            raise DLPyError('image_id should be an integer between 0'
                            ' and {}.'.format(uid.shape[0] - 1))
        uid_value = uid.iloc[idx, 1]
        uid_name = uid.columns[1]

        input_tbl = input_table_check(data)

        feature_maps_tbl = random_name('Feature_Maps') + '_{}'.format(idx)
        score_options = dict(model=self.model_table, initWeights=self.model_weights,
                             table=dict(where='{}="{}"'.format(uid_name,
                                                               uid_value), **input_tbl),
                             layerOut=dict(name=feature_maps_tbl),
                             randomflip='none',
                             randomcrop='none',
                             layerImageType='jpg',
                             encodeName=True)
        score_options.update(kwargs)
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
        """
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

        """
        input_tbl_opts = input_table_check(data)
        input_table = self.conn.CASTable(**input_tbl_opts)
        if target not in input_table.columns.tolist():
            raise DLPyError('Column name "{}" not found in the data table.'.format(target))

        feature_tbl = random_name('Features')
        score_options = dict(model=self.model_table, initWeights=self.model_weights,
                             table=dict(**input_tbl_opts),
                             layerOut=dict(name=feature_tbl),
                             layerList=dense_layer,
                             layerImageType='wide',
                             randomflip='none',
                             randomcrop='none',
                             encodeName=True)
        score_options.update(kwargs)
        self._retrieve_('deeplearn.dlscore', **score_options)
        x = self.conn.CASTable(feature_tbl).as_matrix()
        y = self.conn.CASTable(**input_tbl_opts)[target].as_matrix().ravel()
        return x, y

    def heat_map_analysis(self, data=None, mask_width=None, mask_height=None, step_size=None,
                          display=True, img_type='A', image_id=None, filename=None, inputs="_image_",
                          target="_label_", max_display=5, **kwargs):
        """
        Conduct a heat map analysis on table of images

        Parameters
        ----------
        data : ImageTable, optional
            If data is None then the results from model.predict are used.
            data specifies the table containing the image data which must contain
            the columns '_image_', '_label_', '_id_' and '_filename_0'.
        mask_width : int, optional
            Specifies the width of the mask which cover the region of the image.
        mask_height : int, optional
            Specifies the height of the mask which cover the region of the image.
        step_size : int, optional
            Specifies the step size of the movement of the the mask.
        display : bool, optional
            Specifies whether to display the results.
        img_type : string, optional
            Can be 'A' for all images, 'C' for only correctly classified images, or
            'M' for misclassified images.
        image_id : list or int, optional
            A unique image id to get the heatmap. A standard column of ImageTable
        filename : list of strings or string, optional
            The name of a file in '_filename_0' if not unique returns multiple
        inputs : string, optional
            Name of image column for the input into the model.predict function
        target : string, optional
            Name of column for the correct label
        max_display : int, optional
            Maximum number of images to display. Heatmap takes a significant amount
            of time to run so a max of 5 is default.
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

        """
        def get_predictions(data=data, inputs=inputs, target=target, kwargs=kwargs):
            input_tbl_opts = input_table_check(data)
            input_table = self.conn.CASTable(**input_tbl_opts)
            if target not in input_table.columns.tolist():
                raise DLPyError('Column name "{}" not found in the data table.'.format(target))

            if inputs not in input_table.columns.tolist():
                raise DLPyError('Column name "{}" not found in the data table.'.format(inputs))

            input_table = self.conn.CASTable(**input_tbl_opts)
            input_table = ImageTable.from_table(input_table)
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
            self._retrieve_('deeplearn.dlscore', **dlscore_options_com)
            return self.conn.CASTable(valid_res_tbl_com)

        from .images import ImageTable

        run_predict = True
        if data is None and self.valid_res_tbl is None:
            raise ValueError('No input data and model.predict() has not been run')
        elif data is None:
            print("Using results from model.predict()")
            data = self.valid_res_tbl
            run_predict = False
        elif data.shape[0] == 0:
            raise ValueError('Input table is empty.')

        data = data.partition(casout=dict(name='temp_anotated', replace=True))['casTable']

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

        copy_vars = ImageTable.from_table(data).columns.tolist()

        masked_image_table = random_name('MASKED_IMG')
        blocksize = image_blocksize(output_width, output_height)

        filtered = None
        if filename or image_id:
            print(" filtering by filename or _id_ ")
            if '_id_' not in data.columns.tolist():
                print("'_id_' column not in cas_table, processing complete table")
            else:
                if filename and image_id:
                    print(" image_id supersedes filename, image_id being used")

                if image_id:
                    filtered = filter_by_image_id(data, image_id)
                elif filename:
                    filtered = filter_by_filename(data, filename)

            if filtered:
                self.conn.droptable(data)
                data = filtered

        if run_predict:
            print("Running prediction ...")
            data = get_predictions(data)
            print("... finished running prediction")

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
            data= self.conn.CASTable(sample_tbl)

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

        valid_res_tbl = self.conn.CASTable(valid_res_tbl)

        temp_table = valid_res_tbl.to_frame()
        image_id_list = temp_table['_parentId_'].unique().tolist()
        n_masks = len(temp_table['_id_'].unique())

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
            model_explain_table[name][y:min(y + y_step, output_height), x:min(x + x_step, output_width), count_for_subject[name]] = prob
            count_for_subject[name] += 1

        original_image_table = data.fetchimages(fetchVars=data.columns.tolist(),
                                                to=data.numrows().numrows).Images

        prob_cols = []
        for col in data.columns:
            if 'P__label' in col:
                prob_cols.append(col)

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
                top = bottom + height

                output_str = 'Predicted Label: {}'.format(pred_label)
                output_str += ', filename: {}'.format(filename)
                output_str += ', image_id: {},'.format(id_num)

                axs[im_idx][0].text(left, 0.5 * (bottom + top), output_str,
                                    horizontalalignment='left',
                                    verticalalignment='center',
                                    fontsize=14, color='black',
                                    transform=axs[im_idx][0].transAxes)

            plt.show()

        self.conn.droptable(data)

        return output_table

    def plot_heat_map(self, idx=0, alpha=.2):
        """
        Display the heat maps analysis results

        Displays plot of three images: original, overlayed image and heat map,
        from left to right.

        Parameters
        ----------
        idx : int, optional
            Specifies the image to be displayed, starting from 0.
        alpha : double, optional
            Specifies transparent ratio of the heat map in the overlayed image.
            Must be a numeric between 0 and 1.

        """
        label = self.model_explain_table['_label_'][idx]

        img = self.model_explain_table['_image_'][idx]

        heat_map = self.model_explain_table['heat_map'][idx]

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
        """
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.

        """
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
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        self.save_to_table_with_caslibify(path)
        """dir_name, file_name = os.path.split(path)

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
        print('NOTE: Model table saved successfully.')"""

    def save_to_table_with_caslibify(self, path):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.

        """
        #import os
        #if path.endswith(os.path.sep):
        #    path = path[:-1]

        caslib, path_remaining = caslibify(self.conn, path, task='save')

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = path_remaining + _file_name_ + _extension_
        weight_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
        attr_tbl_file = path_remaining + _file_name_ + '_weights_attr' + _extension_

        if self.model_table is not None:
            ch = self.conn.table.tableexists(self.model_weights)
            if ch.exists > 0:
                rt = self._retrieve_('table.save', table=self.model_table, name=model_tbl_file, replace=True, caslib=caslib)
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

                rt = self._retrieve_('table.save', table=CAS_tbl_name, name=attr_tbl_file, replace=True, caslib=caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model attributes to a table!')

        print('NOTE: Model table saved successfully.')

    def deploy(self, path, output_format='astore', **kwargs):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables or astore
        output_format : string, optional
            Specifies the format of the deployed model
            Valid Values: astore or castable
            Default: astore

        Notes
        -----
        Currently, this function only supports sashdat and astore formats.

        """
        if output_format.lower() == 'astore':
            self.save_to_astore(path=path, **kwargs)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path=path)
        else:
            raise DLPyError('output_format must be "astore", "castable" or "table"')

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

    @property
    def summary(self):
        if self.model_type == 'CNN':
            return pd.concat([x.summary for x in self.layers], ignore_index=True)
        else:
            return pd.concat([x.rnn_summary for x in self.layers], ignore_index=True)

    def print_summary(self):
        ''' Display a table that summarizes the model architecture '''
        try:
            from IPython.display import display

            if self.model_type == 'CNN':
                total_number_of_parameters = 0
                for l in self.layers:
                    if l.num_weights is not None:
                        total_number_of_parameters += l.num_weights
                    if l.num_bias is not None:
                        total_number_of_parameters += l.num_bias

                total = pd.DataFrame([['', '', '', '', '', '', total_number_of_parameters]],
                                     columns=['Layer', 'Type', 'Kernel Size', 'Stride', 'Activation', 'Output Size',
                                              'Number of Parameters'])
                display(pd.concat([self.summary, total], ignore_index=True))
            else:
                display(self.summary)

        except ImportError:
            print(self.summary)

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


class FeatureMaps(object):
    '''
    Feature Maps object

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    feature_maps_tbl : CAS table
        Specifies the CAS table to store the feature maps.
    structure : dict, optional
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
        filter_id : list-of-ints, optional
            Specifies the filters to be displayed.
            Default: None

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
            plt.tight_layout(pad=2.5, rect=[0, 0.03, 1, 0.95])
        plt.show()


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
    dict
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
    dict
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
    dict
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
    dict
        Options that can be passed to layer definition

    '''
    num_keys = ['n_filters', 'width', 'height', 'stride', 'std', 'mean',
                'init_bias', 'dropout', 'truncation_factor', 'init_b', 'trunc_fact']
    str_keys = ['act', 'init']

    conv_layer_config = dict()
    conv_layer_config.update(get_num_configs(num_keys, 'convopts', layer_table))
    conv_layer_config.update(get_str_configs(str_keys, 'convopts', layer_table))
    conv_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    if 'trunc_fact' in conv_layer_config.keys():
        conv_layer_config['truncation_factor'] = conv_layer_config['trunc_fact']
        del conv_layer_config['trunc_fact']

    if layer_table['_DLNumVal_'][layer_table['_DLKey1_'] == 'convopts.no_bias'].any():
        conv_layer_config['include_bias'] = False
    else:
        conv_layer_config['include_bias'] = True

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
    dict
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
    dict
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
    dict
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
    dict
        Options that can be passed to layer definition

    '''

    concat_layer_config = dict()

    concat_layer_config.update(get_str_configs(['act'], 'residualopts', layer_table))
    concat_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]

    layer = Concat(**concat_layer_config)
    return layer


def extract_detection_layer(layer_table):
    detection_layer_config = dict()

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
    dict
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

    if 'trunc_fact' in fc_layer_config.keys():
        fc_layer_config['truncation_factor'] = fc_layer_config['trunc_fact']
        del fc_layer_config['trunc_fact']

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
    dict
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

    if 'trunc_fact' in output_layer_config.keys():
        output_layer_config['truncation_factor'] = output_layer_config['trunc_fact']
        del output_layer_config['trunc_fact']

    layer = OutputLayer(**output_layer_config)
    return layer


def extract_keypoints_layer(layer_table):
    # TODO
    keypoints_layer_config = dict()
    keypoints_layer_config['name'] = layer_table['_DLKey0_'].unique()[0]
    layer = Keypoints(**keypoints_layer_config)
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
    dict
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
    dict
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


class DLPyDict(collections.MutableMapping):
    """ Dictionary that applies an arbitrary key-altering function before accessing the keys """

    def __init__(self, *args, **kwargs):
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __getitem__(self, key):
        return self.__dict__[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if value is not None:
            self.__dict__[self.__keytransform__(key)] = value
        else:
            if key in self.__dict__:
                self.__delitem__[key]

    def __delitem__(self, key):
        del self.__dict__[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __keytransform__(self, key):
        return key.lower().replace("_", "")

    def __str__(self):
        return str(self.__dict__)


class Solver(DLPyDict):
    '''
    Solver object

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously with
        multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are less
        than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches one
        of the specified steps, the learning rate is multiplied by the value
        of the gamma parameter. For example, if you specify {5, 9, 13}, then
        the learning rate is multiplied by gamma after the fifth, ninth, and
        thirteenth epochs.

    Returns
    -------
    :class:`Solver`

    '''
    def __init__(self, learning_rate=0.001, learning_rate_policy='fixed', gamma=0.1, step_size=10, power=0.75,
                 use_locking=True, clip_grad_max=None, clip_grad_min=None, steps=None):
        DLPyDict.__init__(self, learning_rate=learning_rate, learning_rate_policy=learning_rate_policy, gamma=gamma,
                          step_size=step_size, power=power, use_locking=use_locking, clip_grad_max=clip_grad_max,
                          clip_grad_min=clip_grad_min, steps=steps)

    def set_method(self, method):
        '''
        Sets the solver method in the parameters list.

        Parameters
        ----------
        method : string
            Specifies the type of the solver method.
            Possible values: ['vanilla', 'momentum', 'adam', 'lbfg', 'natgrad']
        '''
        self.add_parameter('method', method)

    def add_parameter(self, key, value):
        '''
        Adds a parameter to the parameter list of a solver.

        Parameters
        ---------
        key : string
            Specifies the name of the parameter to be added to the list
        value : string
            Specifies the actual values of the parameter to be added to the list
        '''
        self.__setitem__(key, value)

    def __str__(self):
        return super().__str__()


class VanillaSolver(Solver):
    '''
    Vanilla solver object

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously with
        multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are less
        than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches
        one of the specified steps, the learning rate is multiplied by the
        value of the gamma parameter. For example, if you specify {5, 9, 13},
        then the learning rate is multiplied by gamma after the fifth, ninth,
        and thirteenth epochs.

    Returns
    -------
    :class:`VanillaSolver`

    '''
    def __init__(self, learning_rate=0.001, learning_rate_policy='fixed', gamma=0.1, step_size=10, power=0.75,
                 use_locking=True, clip_grad_max=None, clip_grad_min=None, steps=None):
        Solver.__init__(self, learning_rate, learning_rate_policy, gamma, step_size, power, use_locking,
                        clip_grad_max, clip_grad_min, steps)
        self.set_method('vanilla')


class MomentumSolver(Solver):
    '''
    Momentum solver object

    Parameters
    -----------
    momentum : double, optional
        Specifies the momentum for stochastic gradient descent.
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously
        with multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are
        less than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches
        one of the specified steps, the learning rate is multiplied by the
        value of the gamma parameter. For example, if you specify {5, 9, 13},
        then the learning rate is multiplied by gamma after the fifth,
        ninth, and thirteenth epochs.

    Returns
    -------
    :class:`MomentumSolver`

    '''
    def __init__(self, momentum=0.9, learning_rate=0.001, learning_rate_policy='fixed', gamma=0.1, step_size=10,
                 power=0.75, use_locking=True, clip_grad_max=None, clip_grad_min=None, steps=None):
        Solver.__init__(self, learning_rate, learning_rate_policy, gamma, step_size, power, use_locking,
                        clip_grad_max, clip_grad_min, steps)
        self.set_method('momentum')
        self.add_parameter('momentum', momentum)


class AdamSolver(Solver):
    '''
    Adam solver object

    Parameters
    ----------
    beta1 : double, optional
        Specifies the exponential decay rate for the first moment in
        the Adam learning algorithm.
    beta2 : double, optional
        Specifies the exponential decay rate for the second moment in
        the Adam learning algorithm.
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size: int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously with
        multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are less
        than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches
        one of the specified steps, the learning rate is multiplied by the
        value of the gamma parameter. For example, if you specify {5, 9, 13},
        then the learning rate is multiplied by gamma after the fifth, ninth,
        and thirteenth epochs.

    Returns
    -------
    :class:`AdamSolver`

    '''
    def __init__(self, beta1=0.9, beta2=0.999, learning_rate=0.001, learning_rate_policy='fixed', gamma=0.1,
                 step_size=10, power=0.75, use_locking=True, clip_grad_max=None, clip_grad_min=None, steps=None):
        Solver.__init__(self, learning_rate, learning_rate_policy, gamma, step_size, power, use_locking,
                        clip_grad_max, clip_grad_min, steps)
        self.set_method('adam')
        self.add_parameter('beta1', beta1)
        self.add_parameter('beta2', beta2)


class LBFGSolver(Solver):
    '''
    LBFG solver object

    Parameters
    ----------
    m : int
        Specifies the number of corrections used in the L-BFGS update.
    max_line_search_iters : int
        Specifies the maximum number of line search iterations for
        L-BFGS solver.
    max_iters : int
        Specifies the maximum number of iterations for the L-BFGS solver.
        When the miniBatchSize option is not specified, each iteration
        goes through at least one epoch. When the miniBatchSize option is
        specified, each L-BFGS iteration processes one mini-batch.
        The L-BFGS solver stops when the iteration number reaches the value
        of the maxIters= option or the epoch number reaches the value of
        the maxEpochs= option.
    backtrack_ratio : double
        Specifies the backtrack ratio of line search iterations for L-BFGS solver.
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously with
        multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are less
        than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches one
        of the specified steps, the learning rate is multiplied by the value
        of the gamma parameter. For example, if you specify {5, 9, 13}, then
        the learning rate is multiplied by gamma after the fifth, ninth, and
        thirteenth epochs.

    Returns
    -------
    :class:`LBFGSolver`

    '''
    def __init__(self, m, max_line_search_iters, max_iters, backtrack_ratio, learning_rate=0.001,
                 learning_rate_policy='fixed', gamma=0.1, step_size=10, power=0.75, use_locking=True,
                 clip_grad_max=None, clip_grad_min=None, steps=None):
        Solver.__init__(self, learning_rate, learning_rate_policy, gamma, step_size, power, use_locking,
                        clip_grad_max, clip_grad_min, steps)
        self.set_method('lbfg')
        self.add_parameters('m', m)
        self.add_parameters('maxlinesearchiters', max_line_search_iters)
        self.add_parameters('maxiters', max_iters)
        self.add_parameters('backtrackratio', backtrack_ratio)


class NatGradSolver(Solver):
    '''
    Natural gradient solver object

    Parameters
    ----------
    approximation_type : int, optional
        Specifies the approximate natural gradient type.
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
        Default: FIXED
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    use_locking : bool, optional
        When it is false, the gradients are computed asynchronously with
        multiple threads.
    clip_grad_max : double, optional
        Specifies the maximum gradient value. All gradients that are greater
        than the specified value are set to the specified value.
    clip_grad_min : double, optional
        Specifies the minimum gradient value. All gradients that are less
        than the specified value are set to the specified value.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches one
        of the specified steps, the learning rate is multiplied by the value
        of the gamma parameter. For example, if you specify {5, 9, 13}, then
        the learning rate is multiplied by gamma after the fifth, ninth, and
        thirteenth epochs.

    Returns
    -------
    :class:`NatGradSolver`

    '''
    def __init__(self, approximation_type=1, learning_rate=0.001, learning_rate_policy='fixed', gamma=0.1,
                 step_size=10, power=0.75, use_locking=True, clip_grad_max=None, clip_grad_min=None, steps=None):
        Solver.__init__(self, learning_rate, learning_rate_policy, gamma, step_size, power, use_locking,
                        clip_grad_max, clip_grad_min, steps)
        self.set_method('natgrad')
        self.add_parameter('approximationtype', approximation_type)


class Optimizer(DLPyDict):
    '''
    Optimizer object

    Parameters
    ----------
    algorithm : Algorithm, optional
        Specifies the deep learning algorithm.
    mini_batch_size : int, optional
        Specifies the number of observations per thread in a mini-batch.
        You can use this parameter to control the number of observations
        that the action uses on each worker for each thread to compute
        the gradient prior to updating the weights. Larger values use more
        memory. When synchronous SGD is used (the default), the total
        mini-batch size is equal to miniBatchSize * number of threads *
        number of workers. When asynchronous SGD is used (by specifying
        the elasticSyncFreq parameter), each worker trains its own local
        model. In this case, the total mini-batch size for each worker is
        miniBatchSize * number of threads.
    seed : double, optional
        Specifies the random number seed for the random number generator
        in SGD. The default value, 0, and negative values indicate to use
        random number streams based on the computer clock. Specify a value
        that is greater than 0 for a reproducible random number sequence.
    max_epochs : int, optional
        Specifies the maximum number of epochs. For SGD with a single-machine
        server or a session that uses one worker on a distributed server,
        one epoch is reached when the action passes through the data one time.
        For a session that uses more than one worker, one epoch is reached
        when all the workers exchange the weights with the controller one time.
        The syncFreq parameter specifies the number of times each worker
        passes through the data before exchanging weights with the controller.
        For L-BFGS with full batch, each L-BFGS iteration might process more
        than one epoch, and final number of epochs might exceed the maximum
        number of epochs.
    reg_l1 : double, optional
        Specifies the weight for the L1 regularization term. By default,
        L1 regularization is not performed and a value of 0 also disables the
        regularization. Begin with small values such as 1e-6. L1 regularization
        can be combined with L2 regularization.
    reg_l2 : double, optional
        Specifies the weight for the L2 regularization term. By default,
        L2 regularization is not performed and a value of 0 also disables the
        regularization. Begin with small
        values such as 1e-3. L1 regularization can be combined with
        L2 regularization.
    dropout : double, optional
        Specifies the probability that the output of a neuron in a fully
        connected layer will be set to zero during training. The specified
        probability is recalculated each time an observation is processed.
    dropout_input : double, optional
        Specifies the probability that an input variable will be set to zero
        during training. The specified probability is recalculated each time
        an observation is processed.
    dropout_type : string, optional
        Specifies what type of dropout to use.
        Valid Values: STANDARD, INVERTED
        Default: STANDARD
    stagnation : int, optional
        Specifies the number of successive iterations without improvement
        before stopping the optimization early. When the validTable parameter
        is not specified, the loss error is monitored for stagnation. When
        the validTable parameter is specified, the validation scores are
        monitored for stagnation.
    threshold : double, optional
        Specifies the threshold that is used to determine whether the loss
        error or validation score is improving or is stagnating. When
        abs(current_score - previous_score) <= abs(current_score)*threshold,
        the current iteration does not improve the optimization and the
        stagnation counter is incremented. Otherwise, the stagnation counter
        is set to zero.
    f_conv : double, optional
        Specifies the relative function convergence criterion. If the relative
        loss error abs(previous_loss - current_loss) / abs(previous_loss) does
        not result in a change in the objective function, then the optimization
        is stopped. By default, the relative function convergence is not checked.
    snapshot_freq : int, optional
        Specifies the frequency for generating snapshots of the neural weights
        and storing the weights in a weight table during the training process.
        When asynchronous SGD is used, the action synchronizes all the weights
        before writing out the weights.
    log_level : int, optional
        Specifies how progress messages are sent to the client. The default
        value, 0, indicates that no messages are sent. Specify 1 to receive
        start and end messages. Specify 2 to include the iteration history.
    bn_src_layer_warnings : bool, optional
        Turns warning on or off, if batch normalization source layer has
        an atypical type, activation, or include_bias setting. Default: False
        freeze_layers_to : string
        Specifies a layer name to freeze this layer and all the layers before
        this layer.
    total_mini_batch_size : int, optional
        specifies the number of observations in a mini-batch. You can use
        this parameter to control the number of observations that the action
        uses to compute the gradient prior to updating the weights. Larger
        values use more memory. If the specified size cannot be evenly divided
        by the number of threads (if using asynchronous SGD), or the number of
        threads * number of workers (if using synchronous SGD), then the action
        will terminate with an error unless the round parameter was specified
        to be TRUE, in which case, the total mini-batch size will be rounded
        up so that it will be evenly divided.
    flush_weights : bool, optional
        Specifies whether flush the weight table to the disk.
        Default: False
    mini_batch_buf_size : int, optional
        specifies the size of a buffer that is used to save input data and
        intermediate calculations. By default, each layer allocates an input
        buffer that is equal to the number of input channels multiplied by
        the input feature map size multiplied by the bufferSize value. You
        can reduce memory usage by specifying a value that is smaller than
        the bufferSize. The only disadvantage to specifying a small value is
        that run time can increase because multiple smaller matrices must be
        multiplied instead of a single large matrix multiply.

    Returns
    -------
    :class:`Optimizer`

    '''
    def __init__(self, algorithm=VanillaSolver(), mini_batch_size=1, seed=0, max_epochs=1, reg_l1=0, reg_l2=0,
                 dropout=0, dropout_input=0, dropout_type='standard', stagnation=0, threshold=0.00000001, f_conv=0,
                 snapshot_freq=0, log_level=0, bn_src_layer_warnings=True, freeze_layers_to=None, flush_weights=False,
                 total_mini_batch_size=None, mini_batch_buf_size=None):
        DLPyDict.__init__(self, algorithm=algorithm, mini_batch_size=mini_batch_size, seed=seed, max_epochs=max_epochs,
                          reg_l1=reg_l1, reg_l2=reg_l2, dropout=dropout, dropout_input=dropout_input,
                          dropout_type=dropout_type, stagnation=stagnation, threshold=threshold, f_conv=f_conv,
                          snapshot_freq=snapshot_freq, log_level=log_level,
                          bn_src_layer_warnings=bn_src_layer_warnings, freeze_layers_to=freeze_layers_to,
                          flush_weights=flush_weights, total_mini_batch_size=total_mini_batch_size,
                          mini_batch_buf_size=mini_batch_buf_size)

    def add_optimizer_mode(self, solver_mode_type='sync', sync_freq=None, alpha=None, damping=None):
        '''
        Sets the mode of the solver.

        Parameters
        ----------
        solver_mode_type : string
            Specifies the mode of the solver.
        sync_freq : int
            Specifies the synchronization frequency
            This parameter has different details for different solver types:
            For solver_mode_type='sync' and 'downpour'
            specifies the synchronization frequency for SGD in terms of epochs. Set this value to
            0 to use asynchronous SGD.
            For solver_mode_type='elastic'
            Specifies the frequency for communication between the workers and controller for exchanging weights.
            You can exchange weights more often than once each epoch by setting a value that is less than the number of
            batches in an epoch. If this value is greater than the number of batches in an epoch, then the weights
            are exchanged once for each epoch.
        alpha : double
            This parameter should be set only when solver_mode_type='elastic'.
            Specifies the significance level that is used for elastic SGD. When each worker exchanges weights with
            the controller, this value is used to adjust the weights.
        damping : double
            This parameter should be set only when solver_mode_type='elastic'.
            Specifies the damping factor that is used with asynchronous SGD. When each worker exchanges the weights
            with the controller, the weights are combined with this damping factor.
        '''
        mode = {}
        if solver_mode_type == 'downpour':
            mode['type'] = 'downpour'
        elif solver_mode_type == 'elastic':
            mode['type'] = 'elastic'
            if alpha is None:
                mode['alpha'] = 0
            else:
                mode['alpha'] = alpha
            if sync_freq is None:
                mode['syncfreq'] = 0
            else:
                mode['syncfreq'] = sync_freq
            if damping is None:
                mode['damping'] = 0.1
            else:
                mode['damping'] = damping
        else:
            mode['type'] = 'synchronous'
            if sync_freq is None:
                mode['syncfreq'] = 1
            else:
                mode['syncfreq'] = sync_freq
        self.__setitem__('mode', mode)


class TextParms(DLPyDict):
    '''
    Text parameters object

    Parameters
    ----------
    init_input_embeddings : string or CASTable, optional
        specifies an in-memory table that contains the word embeddings.
        By default, the first column is expected to be the terms and
        the rest of the columns are the embedded content.
    init_output_embeddings : string or CASTable, optional
        specifies an in-memory table that contains the word embeddings.
        By default, the first column is expected to be the terms and
        the rest of the columns are the embedded content.
    has_input_term_ids : bool, optional
        Specifies whether the second column of the initial input embedding
        table contains term IDs.
    has_output_term_ids : bool, optional
        Specifies whether the second column of the initial output embedding
        table contains term IDs.
    model_output_embeddings : string or CASTable, optional
        Specifies the output embeddings model table.
    language : string, optional
        Specifies the language for text tokenization.
        Valid Values: ENGLISH, GERMAN, FRENCH, SPANISH, CHINESE, DUTCH,
        FINNISH, ITALIAN, KOREAN, PORTUGUESE, RUSSIAN, TURKISH, JAPANESE,
        POLISH, NORWEGIAN, ARABIC, CZECH, DANISH, INDONESIAN, SWEDISH,
        GREEK, SLOVAK, HEBREW, THAI, VIETNAMESE, SLOVENE, CROATIAN,
        TAGALOG, FARSI, HINDI, HUNGARIAN, ROMANIAN
        default: ENGLISH

    Returns
    -------
    :class:`TextParms`

    '''
    def __init__(self, init_input_embeddings=None, init_output_embeddings=None, has_input_term_ids=False,
                 has_output_term_ids=False, model_output_embeddings=None, language='english'):
        DLPyDict.__init__(self, init_input_embeddings=init_input_embeddings,
                          init_output_embeddings=init_output_embeddings,
                          has_input_term_ids=has_input_term_ids,
                          has_output_term_ids=has_output_term_ids,
                          model_output_embeddings=model_output_embeddings,
                          language=language)


class Sequence(DLPyDict):
    '''
    Sequence parameters object

    Parameters
    ----------
    input_length : string, optional
        This should be a column in the input table.
        Specifies the variable that stores the input sequence length
        (number of tokens) of the row.
    target_length : string, optional
        This should a column / variable in the input table.
        Specifies the variable that stores the target sequence length
        (number of tokens) of the row.
    token_size : int, optional
        Specifies the number of variables that compose one token for
        sequence input data.

    Returns
    -------
    :class:`Sequence`

    '''
    def __init__(self, input_length=None, target_length=None, token_size=1):
        DLPyDict.__init__(self, input_length=input_length, target_length=target_length, token_size=token_size)


class Gpu(DLPyDict):
    '''
    Gpu parameters object.

    Parameters
    ----------
    devices : list-of-ints, optional
        Specifies a list of GPU devices to be used.
    use_tensor_rt : bool, optional
        Enables using TensorRT for fast inference.
        Default: False.
    precision : string, optional
        Specifies the experimental option to incorporate lower computational
        precision in forward-backward computations to potentially engage tensor cores.
        Valid Values: FP32, FP16
        Default: FP32
    use_exclusive : bool, optional
        Specifies exclusive use of GPU devices.
        Default: False

    Returns
    -------
    :class:`Gpu`

    '''
    def __init__(self, devices=None, use_tensor_rt=False, precision='fp32', use_exclusive=False):
        DLPyDict.__init__(self, devices=devices, use_tensor_rt=use_tensor_rt, precision=precision,
                          use_exclusive=use_exclusive)


class DataSpecNumNomOpts(DLPyDict):
    """
    Data spec numeric nominal parameters.

    Parameters
    ----------
    length : string, optional
        Specifies the variable / column that contains the length of the
        data spec input.
    token_size : int, optional
        If positive, data is treated as sequence, else non-sequence

    Returns
    -------
    :class:`DataSpecNumNomOpts`

    """
    def __init__(self, length, token_size=0):
        DLPyDict.__init__(self, length=length, token_size=token_size)


class DataSpec(DLPyDict):
    """
    Data spec parameters.

    Parameters
    -----------
    type_ : string, optional
        Specifies the type of the input data in the data spec.
        Valid Values: NUMERICNOMINAL, NUMNOM, TEXT, IMAGE, OBJECTDETECTION
    layer : string, optional
        Specifies the name of the layer to data spec.
    data : list, optional
        Specifies the name of the columns/variables as the data, this might
        be input or output based on layer type.
    nominals : list, optional
        Specifies the nominal input variables to use in the analysis.
    numeric_nominal_parms : :class:`DataSpecNumNomOpts`, optional
        Specifies the parameters for the numeric nominal data spec inputs.

    Returns
    -------
    :class:`DataSpec`
        A dictionary of data spec parameters.

    """
    def __init__(self, type_, layer, data, nominals=None, numeric_nominal_parms=None):
        DLPyDict.__init__(self, type=type_, layer=layer, data=data, nominals=nominals,
                          numeric_nominal_parms=numeric_nominal_parms)
