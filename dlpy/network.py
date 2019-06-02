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
from dlpy.utils import DLPyError, input_table_check, random_name, check_caslib, caslibify, get_server_path_sep, underscore_to_camelcase
from .layers import InputLayer, Conv2d, Pooling, BN, Res, Concat, Dense, OutputLayer, Keypoints, Detection, Scale,\
    Reshape, GroupConv2d, ChannelShuffle, RegionProposal, ROIPooling, FastRCNN, Conv2DTranspose, Recurrent
import dlpy.model
import collections
import pandas as pd
import swat as sw
from copy import deepcopy
from . import __dev__


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
        # works for Sequential() as well
        if self.__class__.__name__ == 'Model':
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
        rt = self._retrieve_('deeplearn.buildmodel',
                             model=dict(name=self.model_name, replace=True), type=self.model_type)

        if not all(x.can_be_last_layer for x in self.output_layers):
            raise DLPyError('Output layers can only be {}' \
                            .format([i.__name__ for i in Layer.__subclasses__() if i.can_be_last_layer]))

        if rt.severity > 1:
            raise DLPyError('cannot build model, there seems to be a problem.')
        self.num_params = 0
        for layer in self.layers:
            option = layer.to_model_params()
            rt = self._retrieve_('deeplearn.addlayer', model = self.model_name, **option)
            if rt.severity > 1:
                raise DLPyError('there seems to be an error while adding the ' + layer.name + '.')
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
        copied_model = deepcopy(self)  # deepcopy the sequential model and don't touch the original one
        stop_layers = stop_layers or []
        input_tensors = []
        output_tensors = []

        if not isinstance(stop_layers, collections.Iterable):
            stop_layers = [stop_layers]
        index_l = [self.layers.index(x) for x in stop_layers]
        stop_layers = [copied_model.layers[i] for i in index_l]

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
                    # skip if stop_layers are visited and add its src_layers's output tensors
                    if outbound_layer in stop_layers:
                        for src_layer in outbound_layer.src_layers:
                            output_tensors.append(src_layer.tensor)
                        continue
                    # initialize tensor of the outbound_layer
                    # if any of tensor doesn't exit, stop calling the layer
                    try:
                        outbound_layer([l.tensor for l in outbound_layer.src_layers])
                    except AttributeError:
                        continue
                    if outbound_layer.can_be_last_layer:
                        output_tensors.append(outbound_layer.tensor)

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
    def from_table(cls, input_model_table, display_note = True, output_model_table = None):
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
        model = cls(conn = input_model_table.get_connection(), model_table = output_model_table)
        model_name = model._retrieve_('table.fetch',
                                      table = dict(where = '_DLKey1_= "modeltype"',
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
                model.layers.append(extract_input_layer(layer_table = layer_table))
            elif layer_type == 2:
                model.layers.append(extract_conv_layer(layer_table = layer_table))
            elif layer_type == 3:
                model.layers.append(extract_pooling_layer(layer_table = layer_table))
            elif layer_type == 4:
                model.layers.append(extract_fc_layer(layer_table = layer_table))
            elif layer_type == 5:
                model.layers.append(extract_output_layer(layer_table = layer_table))
            elif layer_type == 6:
                model.layers.append(extract_recurrent_layer(layer_table = layer_table))
            elif layer_type == 8:
                model.layers.append(extract_batchnorm_layer(layer_table = layer_table))
            elif layer_type == 9:
                model.layers.append(extract_residual_layer(layer_table = layer_table))
            elif layer_type == 10:
                model.layers.append(extract_concatenate_layer(layer_table = layer_table))
            elif layer_type == 11:
                model.layers.append(extract_detection_layer(layer_table = layer_table))
            elif layer_type == 12:
                model.layers.append(extract_scale_layer(layer_table=layer_table))
            elif layer_type == 13:
                model.layers.append(extract_keypoints_layer(layer_table = layer_table))
            elif layer_type == 14:
                model.layers.append(extract_reshape_layer(layer_table = layer_table))
            elif layer_type == 16:
                model.layers.append(extract_conv2dtranspose_layer(layer_table = layer_table))
            elif layer_type == 17:
                model.layers.append(extract_groupconv_layer(layer_table = layer_table))
            elif layer_type == 18:
                model.layers.append(extract_channelshuffle_layer(layer_table = layer_table))
            elif layer_type == 23:
                model.layers.append(extract_rpn_layer(layer_table = layer_table))
            elif layer_type == 24:
                model.layers.append(extract_roipooling_layer(layer_table = layer_table))
            elif layer_type == 25:
                model.layers.append(extract_fastrcnn_layer(layer_table = layer_table))

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
    def from_sashdat(cls, conn, path, output_model_table = None):
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
        model = cls(conn, model_table = output_model_table)
        model.load(path = path)
        return model

    @classmethod
    def from_caffe_model(cls, conn, input_network_file, output_model_table = None,
                         model_weights_file = None, **kwargs):
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
            output_model_table = dict(name = random_name('caffe_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name = random_name('caffe_model', 6)))

        model_name = model_table_opts['name']

        output_code = caffe_to_sas(input_network_file, model_name, network_param = model_weights_file, **kwargs)
        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)
        return model

    @classmethod
    def from_keras_model(cls, conn, keras_model, output_model_table = None,
                         offsets=None, std=None, scale=1.0,
                         max_num_frames=-1, include_weights = False,
                         input_weights_file = None, verbose=False):
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
            output_model_table = dict(name = random_name('keras_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name = random_name('keras_model', 6)))

        model_name = model_table_opts['name']

        # determine what features are supported by current Viya server/deep learning action set
        from .model_conversion.model_conversion_utils import check_rnn_import, check_normstd
        rnn_support = check_rnn_import(conn)
        normstd_support = check_normstd(conn)
        if (std is not None) and (not normstd_support):
            print('WARNING: Your Viya installation does not support the std parameter - ignoring')
            std = None

        output_code = keras_to_sas(model = keras_model, rnn_support = rnn_support,
                                   model_name = model_name, offsets = offsets, std = std,
                                   scale = scale, max_num_frames = max_num_frames, verbose = verbose)

        if verbose:
            print(output_code)

        exec(output_code)
        temp_name = conn
        exec('sas_model_gen(temp_name)')
        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)

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
    def from_onnx_model(cls, conn, onnx_model, output_model_table = None,
                        offsets = None, scale = None, std = None, output_layer = None):
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
            output_model_table = dict(name = random_name('onnx_model', 6))

        model_table_opts = input_table_check(output_model_table)

        if 'name' not in model_table_opts:
            model_table_opts.update(**dict(name = random_name('onnx_model', 6)))

        model_name = model_table_opts['name']

        _layers = onnx_to_sas(onnx_model, model_name, output_layer)
        if offsets is not None:
            _layers[0].config.update(offsets = offsets)
        if scale is not None:
            _layers[0].config.update(scale = scale)
        if std is not None:
            _layers[0].config.update(std = std)
        if len(_layers) == 0:
            raise DLPyError('Unable to import ONNX model.')

        conn.loadactionset('deeplearn', _messagelevel = 'error')
        rt = conn.retrieve('deeplearn.buildmodel',
                           _messagelevel = 'error',
                           model = dict(name = model_name, replace = True),
                           type = 'CNN')
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('Cannot build model, there seems to be a problem.')

        for layer in _layers:
            option = layer.to_model_params()
            rt = conn.retrieve('deeplearn.addlayer', _messagelevel = 'error',
                               model = model_name, **option)
            if rt.severity > 1:
                for m in rt.messages:
                    print(m)
                raise DLPyError('There seems to be an error while adding the '
                                + layer.name + '.')

        input_model_table = conn.CASTable(**model_table_opts)
        model = cls.from_table(input_model_table = input_model_table)
        print('NOTE: Successfully imported ONNX model.')
        return model

    @property
    def summary(self):
        if self.model_type == 'CNN':
            return pd.concat([x.summary for x in self.layers], ignore_index = True)
        else:
            return pd.concat([x.rnn_summary for x in self.layers], ignore_index = True)

    def __load_layer_ids(self):
        try:
            model_table_rows = self.conn.table.fetch(self.model_table, maxrows = 1000000, to = 1000000).Fetch
        except:
            model_table_rows = None
        if model_table_rows is not None:
            layer_ids = {}
            import math
            for index, row in model_table_rows.iterrows():
                if not math.isnan(row['_DLLayerID_']):
                    layer_ids[row['_DLKey0_']] = int(row['_DLLayerID_'])

            for l in self.layers:
                l.layer_id = layer_ids[l.name.lower()]

    def print_summary(self):
        ''' Display a table that summarizes the model architecture '''
        try:
            if len(self.layers) > 0 and self.layers[0].layer_id is None:
                self.__load_layer_ids()

            from IPython.display import display

            if self.model_type == 'CNN':
                if self.num_params is None:
                    self.num_params = 0
                    for l in self.layers:
                        if l.num_weights is not None:
                            self.num_params += l.num_weights
                        if l.num_bias is not None:
                            self.num_params += l.num_bias

                total = pd.DataFrame([['', '', '', '', '', '', '', self.num_params]],
                                     columns=['Layer Id', 'Layer', 'Type', 'Kernel Size', 'Stride', 'Activation',
                                              'Output Size', 'Number of Parameters'])
                display(pd.concat([self.summary, total], ignore_index = True))
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

        cas_lib_name, file_name, tmp_caslib = caslibify(self.conn, path, task='load')

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
                model.layers.append(extract_recurrent_layer(layer_table = layer_table))
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
                self.layers.append(extract_keypoints_layer(layer_table = layer_table))
            elif layer_type == 14:
                self.layers.append(extract_reshape_layer(layer_table = layer_table))
            elif layer_type == 16:
                self.layers.append(extract_conv2dtranspose_layer(layer_table = layer_table))
            elif layer_type == 17:
                self.layers.append(extract_groupconv_layer(layer_table = layer_table))
            elif layer_type == 18:
                self.layers.append(extract_channelshuffle_layer(layer_table = layer_table))
            elif layer_type == 23:
                self.layers.append(extract_rpn_layer(layer_table = layer_table))
            elif layer_type == 24:
                self.layers.append(extract_roipooling_layer(layer_table = layer_table))
            elif layer_type == 25:
                self.layers.append(extract_fastrcnn_layer(layer_table = layer_table))

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

        if (cas_lib_name is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights(self, path, labels=False, data_spec=None, label_file_name=None, label_length=None,
                     use_gpu=False):
        '''
        Load the weights form a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.
        labels : bool
            Specifies whether to apply user-defined classification labels
        data_spec : list of :class:`DataSpec`, optional
            data specification for input and output layer(s)
        label_file_name : string, optional
            Fully qualified path to CSV file containing user-defined
            classification labels.  If not specified, ImageNet labels assumed.
        label_length : int, optional
            Length of the classification labels (in characters).
        use_gpu: boolean, optional
            GPU processing of model required (or not)

        Notes
        -----
        Currently support HDF5 and sashdat files.

        '''

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
                                         label_length=label_length, use_gpu=use_gpu)
        elif file_name.lower().endswith('onnxmodel.h5'):
            self.load_weights_from_keras(path, labels=labels, data_spec=data_spec, label_file_name=label_file_name,            
                                         label_length=label_length, use_gpu=use_gpu)
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
                                use_gpu=False):
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
        use_gpu : boolean, optional
            Require GPU for processing model

        '''
        if labels:
            self.load_weights_from_file_with_labels(path=path, format_type='KERAS', data_spec=data_spec,
                                                    label_file_name=label_file_name, label_length=label_length,
                                                    use_gpu=use_gpu)
        else:
            self.load_weights_from_file(path=path, format_type='KERAS', data_spec=data_spec, use_gpu=use_gpu)

    def load_weights_from_file(self, path, format_type='KERAS', data_spec=None, use_gpu=False):
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
        use_gpu : boolean, optional
            Require GPU for processing model

        '''
        cas_lib_name, file_name, tmp_caslib = caslibify(self.conn, path, task='load')

        if data_spec:

            # run action with dataSpec option
            with sw.option_context(print_messages = False):
                rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                    model=self.model_table,
                                    modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                    dataSpecs=data_spec,
                                    gpuModel=use_gpu,
                                    formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                    );

            # if error, may not support dataspec
            if rt.severity > 1:

                # check for error containing "dataSpecs"
                data_spec_missing = False
                for msg in rt.messages:
                    if ('ERROR' in msg) and ('dataSpecs' in msg):
                        data_spec_missing = True

                if data_spec_missing:
                    with sw.option_context(print_messages = False):
                        rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                            modelWeights=dict(replace=True,
                                                              name=self.model_name + '_weights'),
                                            formatType=format_type, weightFilePath=file_name,
                                            gpuModel=use_gpu,
                                            caslib=cas_lib_name,
                                            )

                # handle error or create necessary attributes
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('Cannot import model weights, there seems to be a problem.')
                else:
                    from dlpy.attribute_utils import create_extended_attributes
                    create_extended_attributes(self.conn, self.model_name, self.layers, data_spec)

        else:
            print("NOTE: no dataspec(s) provided - creating image classification model.")
            self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                            modelWeights=dict(replace=True,
                                              name=self.model_name + '_weights'),
                            formatType=format_type, weightFilePath=file_name,
                            gpuModel=use_gpu,
                            caslib=cas_lib_name,
                            )

        self.set_weights(self.model_name + '_weights')

        if (cas_lib_name is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights_from_file_with_labels(self, path, format_type='KERAS', data_spec=None, label_file_name=None, label_length=None,
                                           use_gpu=False):
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

        '''
        cas_lib_name, file_name, tmp_caslib = caslibify(self.conn, path, task='load')

        if (label_file_name):
            from dlpy.utils import get_user_defined_labels_table
            label_table = get_user_defined_labels_table(self.conn, label_file_name, label_length)
        else:
            from dlpy.utils import get_imagenet_labels_table
            label_table = get_imagenet_labels_table(self.conn, label_length)            

        if (data_spec):

            # run action with dataSpec option
            with sw.option_context(print_messages = False):
                rt = self._retrieve_('deeplearn.dlimportmodelweights',
                                    model=self.model_table,
                                    modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                    dataSpecs=data_spec,
                                    gpuModel=use_gpu,
                                    formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                    labelTable=label_table,
                                    );

            # if error, may not support dataspec
            if rt.severity > 1:

                # check for error containing "dataSpecs"
                data_spec_missing = False
                for msg in rt.messages:
                    if ('ERROR' in msg) and ('dataSpecs' in msg):
                        data_spec_missing = True

                if data_spec_missing:
                    with sw.option_context(print_messages = False):
                        rt = self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                                            modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                                            formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                                            gpuModel=use_gpu,
                                            labelTable=label_table,
                                            );

                # handle error or create necessary attributes with Python function
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('Cannot import model weights, there seems to be a problem.')
                else:
                    from dlpy.attribute_utils import create_extended_attributes
                    create_extended_attributes(self.conn, self.model_name, self.layers, data_spec, label_file_name)

        else:
            print("NOTE: no dataspec(s) provided - creating image classification model.")
            self._retrieve_('deeplearn.dlimportmodelweights', model=self.model_table,
                            modelWeights=dict(replace=True, name=self.model_name + '_weights'),
                            formatType=format_type, weightFilePath=file_name, caslib=cas_lib_name,
                            gpuModel=use_gpu,
                            labelTable=label_table,
                            );

        self.set_weights(self.model_name + '_weights')

        if (cas_lib_name is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

    def load_weights_from_table(self, path):
        '''
        Load the weights from a file

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.

        '''
        cas_lib_name, file_name, tmp_caslib = caslibify(self.conn, path, task='load')

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

        if (cas_lib_name is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = cas_lib_name)

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
        
        if os.path.isfile(path):
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
                if isinstance(shares, str):
                    shares = [shares]
                for share in shares:
                    idx_share = layers_name.index(share)
                    self.layers[idx_share].shared_weights = anchor

    def save_to_astore(self, path = None, **kwargs):
        """
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.

        """
        self.conn.loadactionset('astore', _messagelevel = 'error')

        CAS_tbl_name = self.model_name + '_astore'

        self._retrieve_('deeplearn.dlexportmodel',
                        casout = dict(replace = True, name = CAS_tbl_name),
                        initWeights = self.model_weights,
                        modelTable = self.model_table,
                        randomCrop = 'none',
                        randomFlip = 'none',
                        randomMutation = 'none',
                        **kwargs)

        model_astore = self._retrieve_('astore.download',
                                       rstore = CAS_tbl_name)

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

        caslib, path_remaining, tmp_caslib = caslibify(self.conn, path, task = 'save')

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = path_remaining + _file_name_ + _extension_
        weight_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
        attr_tbl_file = path_remaining + _file_name_ + '_weights_attr' + _extension_

        if self.model_table is not None:
            ch = self.conn.table.tableexists(self.model_weights)
            if ch.exists > 0:
                rt = self._retrieve_('table.save', table = self.model_table, name = model_tbl_file, replace = True,
                                     caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model to a table!')
        if self.model_weights is not None:
            ch = self.conn.table.tableexists(self.model_weights)
            if ch.exists > 0:
                rt = self._retrieve_('table.save', table = self.model_weights, name = weight_tbl_file,
                                     replace = True, caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model weights to a table!')

                CAS_tbl_name = random_name('Attr_Tbl')
                rt = self._retrieve_('table.attribute', task = 'convert', attrtable = CAS_tbl_name,
                                     **self.model_weights.to_table_params())
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while extracting the model attributes!')

                rt = self._retrieve_('table.save', table = CAS_tbl_name, name = attr_tbl_file, replace = True,
                                     caslib = caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model attributes to a table!')

        print('NOTE: Model table saved successfully.')

        if (caslib is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = caslib)

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
        weights_table_opts.update(**dict(groupBy = '_LayerID_',
                                         groupByMode = 'REDISTRIBUTE',
                                         orderBy = '_WeightID_'))
        self.conn.partition(table = weights_table_opts,
                            casout = dict(name = self.model_weights.name,
                                          replace = True))

        caslib, path_remaining, tmp_caslib = caslibify(self.conn, path, task = 'save')
        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.csv'
        weights_tbl_file = path_remaining + _file_name_ + '_weights' + _extension_
        rt = self._retrieve_('table.save', table = weights_table_opts,
                             name = weights_tbl_file, replace = True, caslib = caslib)
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('something is wrong while saving the the model to a table!')

        print('NOTE: Model weights csv saved successfully.')

        if (caslib is not None) and tmp_caslib:
            self._retrieve_('table.dropcaslib', message_level = 'error', caslib = caslib)

    def save_to_onnx(self, path, model_weights = None):
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
                                 model_table = model_table,
                                 model_weights = model_weights)
        file_name = self.model_name + '.onnx'
        if path is None:
            path = os.getcwd()

        if not os.path.isdir(path):
            os.makedirs(path)

        file_name = os.path.join(path, file_name)

        with open(file_name, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print('NOTE: ONNX model file saved successfully.')

    def deploy(self, path, output_format = 'astore', model_weights = None, **kwargs):
        """
        Deploy the deep learning model to a data file

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model files.
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
            self.save_to_astore(path = path, **kwargs)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path = path)
        elif output_format.lower() == 'onnx':
            self.save_to_onnx(path, model_weights = model_weights)
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

    return dict(name = layer.name, label = ' %s ' % label,
                fillcolor = bg, color = fg, margin = '0.2,0.0', height = '0.3')


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
        gv_params.append(dict(label = label, tail_name = '{}'.format(item.name),
                              head_name = '{}'.format(layer.name)))

    if layer.type == 'recurrent':
        gv_params.append(dict(label = '', tail_name = '{}'.format(layer.name),
                              head_name = '{}'.format(layer.name)))
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

    model_graph = gv.Digraph(name = model.model_name,
                             node_attr = dict(shape = 'record', style = 'filled', fontname = 'helvetica'),
                             edge_attr = dict(fontname = 'helvetica', fontsize = '10'))
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
                model_graph.edge(color = '#5677F3', **gv_param)

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
    if conv_layer_config.get('act') == 'Leaky Activation function':
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
    if 'poolingopts.pad_left' in layer_table['_DLKey1_'].values and 'poolingopts.pad_top' in layer_table['_DLKey1_'].values:
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
    if bn_layer_config.get('act') == 'Leaky Activation function':
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
    for i in range(int(predictions_per_grid*2)):
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
    if fc_layer_config.get('act') == 'Leaky Activation function':
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
    if grpconv_layer_config.get('act') == 'Leaky Activation function':
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
    if conv2dtranspose_layer_config.get('act') == 'Leaky Activation function':
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
