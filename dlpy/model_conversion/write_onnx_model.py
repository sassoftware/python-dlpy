#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' Write ONNX model '''

from onnx import defs
from onnx import helper, numpy_helper
from onnx import TensorProto
import numpy as np


class OnnxWriteError(ValueError):
    '''
    Used to indicate an error in parsing ONNX model definition

    '''


def sas_to_onnx(layers, model_table, model_weights):
    '''
    Convert DLPy model to ONNX

    Parameters
    ----------
    layers : iter-of-Layers
        Specifies the layers defining the model.
    model_table : :class:`CASTable`
        Specifies the CASTable of the model.
    model_weights : :class:`pandas.DataFrame` or :class:`CASTable`
        DataFrame or CASTable containing the model weights.
        If this is a CASTable, the weights will be fetched from
        the CAS server.  This may take a long time if 
        the model has many weights.

    Returns
    -------
    Loaded in-memory ModelProto

    '''
    nodes = []
    inputs = []
    outputs = []
    initializer = []
    
    import pandas as pd
    if isinstance(model_weights, pd.DataFrame):
        fetch = False
    else:
        fetch = True

    model_name = model_table.query('_DLKey1_ = "modeltype"') \
                            .fetch()['Fetch']['_DLKey0_'][0]

    for layer in layers:
        if layer.type == 'input':
            H = int(layer.config['height'])
            W = int(layer.config['width'])
            C = int(layer.config['n_channels'])
            value_info = helper.make_tensor_value_info(name=layer.name,
                                                       elem_type=TensorProto.FLOAT,
                                                       shape=[1, C, H, W])
            inputs.append(value_info)

        elif layer.type == 'convo':
            H = int(layer.config['height'])
            W = int(layer.config['width'])
            M = int(layer.config['n_filters'])
            # set stride
            S_h, S_w = get_strides(layer)
            # set padding
            padding = get_padding(layer)

            bias = layer.config['include_bias']
            if bias is None:
                bias = True
            dropout = layer.config['dropout']
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'RECTIFIER'

            # inputs to conv op
            conv_input = [l.name for l in layer.src_layers]
            conv_input.append(layer.name + '_w')
            if bias:
                conv_input.append(layer.name + '_b')

            # create names of node input/output
            if not dropout and act.lower() == 'identity':
                conv_output = [layer.name]
            elif not dropout:
                conv_output = [layer.name + '_conv_out']
                act_input = conv_output
                act_output = [layer.name]
            elif dropout and act.lower() == 'identity':
                conv_output = [layer.name + '_conv_out']
                dropout_input = conv_output
                dropout_output = [layer.name]
            else:
                conv_output = [layer.name + '_conv_out']
                act_input = conv_output
                act_output = [layer.name + '_act_out']
                dropout_input = act_output
                dropout_output = [layer.name]

            conv_op = helper.make_node(op_type='Conv',
                                       inputs=conv_input,
                                       outputs=conv_output,
                                       pads=padding,
                                       kernel_shape=[H, W],
                                       strides=[S_h, S_w])
            nodes.append(conv_op)

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

            # dropout op
            if dropout:
                dropout_op = helper.make_node(op_type='Dropout',
                                              inputs=dropout_input,
                                              outputs=dropout_output,
                                              ratio=dropout)
                nodes.append(dropout_op)

            # create weight tensors
            layer_id = get_layer_id(model_table, layer.name)
            if fetch:
                weights = fetch_weights(model_weights, layer_id)
            else:
                weights = get_weights_from_dataframe(model_weights, layer_id)
            if bias:
                conv_weights = np.array(weights[:-M], dtype=np.float32)
                bias_weights = np.array(weights[-M:], dtype=np.float32)
            else:
                conv_weights = np.array(weights, dtype=np.float32)
            conv_weights = np.reshape(conv_weights, (M, -1, H, W))
            conv_init = numpy_helper.from_array(conv_weights,
                                                name=layer.name+'_w')
            initializer.append(conv_init)
            # add value info to graph input
            inputs.append(
                helper.make_tensor_value_info(name=layer.name+'_w',
                                              elem_type=TensorProto.FLOAT,
                                              shape=list(conv_weights.shape)))

            if bias:
                bias_init = numpy_helper.from_array(bias_weights,
                                                    name=layer.name+'_b')
                initializer.append(bias_init)
                # add value info to graph input
                inputs.append(
                    helper.make_tensor_value_info(name=layer.name+'_b',
                                                  elem_type=TensorProto.FLOAT,
                                                  shape=list(bias_weights.shape)))

        elif layer.type == 'fc':
            n = int(layer.config['n'])
            bias = layer.config['include_bias']
            if bias is None:
                bias = True
            dropout = layer.config['dropout']
            act = layer.config['act']
            if act in [None, 'AUTO']:
               act = 'TANH'

            # inputs to flatten op
            flatten_input = [l.name for l in layer.src_layers]
            flatten_output = [layer.name + '_flatten_out']

            flatten_op = helper.make_node(op_type='Flatten',
                                          inputs=flatten_input,
                                          outputs=flatten_output,
                                          axis=0)
            nodes.append(flatten_op)

            # inputs to fc op (gemm if bias, matmul if no bias)
            fc_input = flatten_output
            fc_input.append(layer.name + '_w')
            if bias:
                fc_input.append(layer.name + '_b')

            # create names of node input/output
            if not dropout and act.lower() == 'identity':
                fc_output = [layer.name]
            elif not dropout:
                fc_output = [layer.name + '_fc_out']
                act_input = fc_output
                act_output = [layer.name]
            elif dropout and act.lower() == 'identity':
                fc_output = [layer.name + '_fc_out']
                dropout_input = fc_output
                dropout_output = [layer.name]
            else:
                fc_output = [layer.name + '_fc_out']
                act_input = fc_output
                act_output = [layer.name + '_act_out']
                dropout_input = act_output
                dropout_output = [layer.name]

            # create fc op
            if bias:
                fc_op = helper.make_node(op_type='Gemm',
                                         inputs=fc_input,
                                         outputs=fc_output)
            else:
                fc_op = helper.make_node(op_type='MatMul',
                                         inputs=fc_input,
                                         outputs=fc_output)
            nodes.append(fc_op)

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

            # dropout op
            if dropout:
                dropout_op = helper.make_node(op_type='Dropout',
                                              inputs=dropout_input,
                                              outputs=dropout_output,
                                              ratio=dropout)
                nodes.append(dropout_op)

            # fc weights
            layer_id = get_layer_id(model_table, layer.name)
            if fetch:
                weights = fetch_weights(model_weights, layer_id)
            else:
                weights = get_weights_from_dataframe(model_weights, layer_id)
            if bias:
                fc_weights = np.array(weights[:-n], dtype=np.float32)
                bias_weights = np.array(weights[-n:], dtype=np.float32)
            else:
                fc_weights = np.array(weights, dtype=np.float32)
            fc_weights = np.reshape(fc_weights, (-1, n))
            fc_init = numpy_helper.from_array(fc_weights,
                                              name=layer.name+'_w')
            initializer.append(fc_init)
            # add value info to inputs
            inputs.append(
                helper.make_tensor_value_info(name=layer.name+'_w',
                                              elem_type=TensorProto.FLOAT,
                                              shape=list(fc_weights.shape)))
            if bias:

                bias_init = numpy_helper.from_array(bias_weights,
                                                    name=layer.name+'_b')

                initializer.append(bias_init)
                # add value info to inputs
                inputs.append(
                    helper.make_tensor_value_info(name=layer.name+'_b',
                                                  elem_type=TensorProto.FLOAT,
                                                  shape=list(bias_weights.shape)))

        elif layer.type == 'pool':
            H = int(layer.config['height'])
            W = int(layer.config['width'])
            # set stride
            S_h, S_w = get_strides(layer)
            # set padding
            padding = get_padding(layer)

            dropout = layer.config['dropout']
            pool = layer.config['pool']

            # create pooling input and output
            pooling_input = [l.name for l in layer.src_layers]
            if dropout:
                pooling_output = [layer.name+'_pool_out']
                dropout_input = pooling_output
                dropout_output = [layer.name]
            else:
                pooling_output = [layer.name]

            # create pooling op
            if pool.lower() == 'max':
                onnx_pool = 'MaxPool'
            elif pool.lower() == 'average' or pool.lower() == 'mean':
                onnx_pool = 'AveragePool'
            else:
                onnx_pool = 'MaxPool'
                print('WARNING: Unsupported pool type '
                      + str(pool) + '. Using MaxPool.')

            pool_op = helper.make_node(op_type=onnx_pool,
                                       inputs=pooling_input,
                                       outputs=pooling_output,
                                       pads=padding,
                                       kernel_shape=[H, W],
                                       strides=[S_h, S_w])
            nodes.append(pool_op)

            # dropout op
            if dropout:
                dropout_op = helper.make_node(op_type='Dropout',
                                              inputs=dropout_input,
                                              outputs=dropout_output,
                                              ratio=dropout)
                nodes.append(dropout_op)

        elif layer.type == 'output':
            # output layer is a loss layer
            if layer.config['full_connect'] == False:
                # get output layer activation
                act = layer.config['act']
                if act in [None, 'AUTO']:
                    act = 'SOFTMAX'

                # create graph output
                if act.lower() == 'identity':
                    output_name = nodes[-1].output[0]
                else:
                    act_input = list(nodes[-1].output)
                    act_output = [layer.name]
                    output_name = layer.name
                    act_op = make_onnx_activation(act, act_input, act_output)
                    nodes.append(act_op)

                # get output dimensions
                dim = layer.src_layers[0].output_size
                if isinstance(dim, int):
                    output_size = [1, dim]
                else:
                    out_w, out_h, out_c = dim
                    output_size = [1, out_c, out_h, out_w]
                # add value info to graph output
                outputs.append(
                    helper.make_tensor_value_info(name=output_name,
                                                  elem_type=TensorProto.FLOAT,
                                                  shape=output_size)
                )
                continue

            n = int(layer.config['n'])
            bias = layer.config['include_bias']
            if bias is None:
                bias = True
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'SOFTMAX'

            # inputs to flatten op
            flatten_input = [l.name for l in layer.src_layers]
            flatten_output = [layer.name + '_flatten_out']

            flatten_op = helper.make_node(op_type='Flatten',
                                          inputs=flatten_input,
                                          outputs=flatten_output,
                                          axis=0)
            nodes.append(flatten_op)

            # inputs to fc op (gemm if bias, matmul if no bias)
            fc_input = flatten_output
            fc_input.append(layer.name + '_w')
            if bias:
                fc_input.append(layer.name + '_b')

            # create names of node input/output
            if act.lower() == 'identity':
                fc_output = [layer.name]
            else:
                fc_output = [layer.name + '_fc_out']
                act_input = fc_output
                act_output = [layer.name]

            # create fc op
            if bias:
                fc_op = helper.make_node(op_type='Gemm',
                                         inputs=fc_input,
                                         outputs=fc_output)
            else:
                fc_op = helper.make_node(op_type='MatMul',
                                         inputs=fc_input,
                                         outputs=fc_output)
            nodes.append(fc_op)

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

            # add output
            outputs.append(
                helper.make_tensor_value_info(name=layer.name,
                                              elem_type=TensorProto.FLOAT,
                                              shape=[1, n]))

            # fc weights
            layer_id = get_layer_id(model_table, layer.name)
            if fetch:
                weights = fetch_weights(model_weights, layer_id)
            else:
                weights = get_weights_from_dataframe(model_weights, layer_id)
            if bias:
                fc_weights = np.array(weights[:-n], dtype=np.float32)
                bias_weights = np.array(weights[-n:], dtype=np.float32)
            else:
                fc_weights = np.array(weights, dtype=np.float32)
            fc_weights = np.reshape(fc_weights, (-1, n))
            fc_init = numpy_helper.from_array(fc_weights,
                                              name=layer.name+'_w')
            initializer.append(fc_init)
            # add value info to inputs
            inputs.append(
                helper.make_tensor_value_info(name=layer.name+'_w',
                                              elem_type=TensorProto.FLOAT,
                                              shape=list(fc_weights.shape)))
            if bias:

                bias_init = numpy_helper.from_array(bias_weights,
                                                    name=layer.name+'_b')

                initializer.append(bias_init)
                # add value info to inputs
                inputs.append(
                    helper.make_tensor_value_info(name=layer.name+'_b',
                                                  elem_type=TensorProto.FLOAT,
                                                  shape=list(bias_weights.shape)))

        elif layer.type == 'batchnorm':
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'IDENTITY'

            # set input and output
            bn_input = [l.name for l in layer.src_layers]

            param_names = ['_scale', '_bias', '_mean', '_variance']
            bn_input += list(map(lambda x: layer.name+x, param_names))

            if act.lower() != 'identity':
                bn_output = [layer.name + '_bn_out']
                act_input = bn_output
                act_output = [layer.name]
            else:
                bn_output = [layer.name]

            # get bn input dimension
            src = layer.src_layers[0]
            if src.type == 'fc':
                n = int(src.config.get('n'))
            else:
                n = int(src.output_size[2])

            # get weights for bn
            layer_id = get_layer_id(model_table, layer.name)
            if fetch:
                weights = fetch_weights(model_weights, layer_id)
            else:
                weights = get_weights_from_dataframe(model_weights, layer_id)

            # [scale, bias, mean, variance]
            bn_weights = [[weights[i*2] for i in range(n)],
                          [weights[i*2+1] for i in range(n)],
                          [weights[i*2+n*2] for i in range(n)],
                          np.square([weights[i*2+n*2+1] for i in range(n)])]
            # add weights to initializer
            # and value info to input
            for idx, init in enumerate(bn_weights):
                initializer.append(
                    numpy_helper.from_array(np.array(init, dtype=np.float32),
                                            name=bn_input[idx+1]))
                inputs.append(
                    helper.make_tensor_value_info(name=bn_input[idx+1],
                                                  elem_type=TensorProto.FLOAT,
                                                  shape=(n,)))
            # bn op
            nodes.append(
                helper.make_node(op_type='BatchNormalization',
                                 inputs=bn_input,
                                 outputs=bn_output))

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

        elif layer.type == 'residual':
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'IDENTITY'

            res_input = [l.name for l in layer.src_layers]
            if act.lower() != 'identity':
                res_output = [layer.name + '_res_out']
                act_input = res_output
                act_output = [layer.name]
            else:
                res_output = [layer.name]

            # sum op
            nodes.append(
                helper.make_node(op_type='Sum',
                                 inputs=res_input,
                                 outputs=res_output))

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

        elif layer.type == 'concat':
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'IDENTITY'

            # get correct order of concat inputs from model table
            l_conf = model_table[model_table['_DLKey0_'] == layer.name.lower()]
            l_conf = l_conf.fetch()['Fetch']
            concat_order = [l_conf[l_conf['_DLKey1_'] == 'srclayers.' + str(i)]
                            for i in range(len(layer.src_layers))]
            concat_order = [row.iloc[0][2] for row in concat_order]
            # concat_order contains lower case layer names
            # sort the names of src layer objects according to this order
            concat_input = [l.name for l in layer.src_layers]
            concat_input = sorted(concat_input,
                                  key=lambda name: concat_order.index(name.lower()))

            if act.lower() != 'identity':
                concat_output = [layer.name + '_concat_out']
                act_input = concat_output
                act_output = [layer.name]
            else:
                concat_output = [layer.name]

            # concat op
            nodes.append(
                helper.make_node(op_type='Concat',
                                 inputs=concat_input,
                                 outputs=concat_output,
                                 axis=1))

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

        elif layer.type == 'detection':
            # get output dimensions
            out_w, out_h, out_c = layer.src_layers[0].output_size
            # add value info to graph output
            outputs.append(
                helper.make_tensor_value_info(name=nodes[-1].output[0],
                                              elem_type=TensorProto.FLOAT,
                                              shape=[1, out_c, out_h, out_w])
            )

        elif layer.type == 'reshape':
            act = layer.config['act']
            if act in [None, 'AUTO']:
                act = 'IDENTITY'

            C = int(layer.config.get('depth'))
            W = int(layer.config.get('width'))
            H = int(layer.config.get('height'))

            reshape_input = [l.name for l in layer.src_layers]

            if act.lower() != 'identity':
                reshape_output = [layer.name + '_reshape_out']
                act_input = reshape_output
                act_output = [layer.name]
            else:
                reshape_output = [layer.name]

            shape = np.array([-1, C, H, W], dtype=np.int64)
            shape_name = layer.name + '_shape'
            shape_init = numpy_helper.from_array(shape,
                                                 name=shape_name)
            initializer.append(shape_init)
            # add value info to inputs
            inputs.append(
                helper.make_tensor_value_info(name=shape_name,
                                              elem_type=TensorProto.INT64,
                                              shape=[4]))

            nodes.append(
                helper.make_node(op_type='Reshape',
                                 inputs=reshape_input+[shape_name],
                                 outputs=reshape_output))

            # activation op
            if act.lower() != 'identity':
                act_op = make_onnx_activation(act, act_input, act_output)
                nodes.append(act_op)

        else:
            layer_type = layer.type
            raise OnnxWriteError(str(layer_type) + ' is not supported.')

    graph_def = helper.make_graph(nodes=nodes,
                                  name=model_name,
                                  inputs=inputs,
                                  outputs=outputs,
                                  initializer=initializer)

    opset = helper.make_opsetid(defs.ONNX_DOMAIN, 8)
    model_def = helper.make_model(graph_def,
                                  producer_name='SAS',
                                  opset_imports=[opset])

    return model_def


def get_layer_id(model_table, layer_name):
    '''
    Get ID of layer from deep learning model
    
    Parameters
    ----------
    model_table : :class:`CASTable`
        CASTable of the deep learning model.
    layer_name : str
        Name of the layer.

    Returns
    -------
    int
        ID of the layer

    '''
    
    return int(model_table.query('_DLKey0_ = "{}"'.format(layer_name.lower())) \
                          .fetch()['Fetch'] \
                          .iloc[0]['_DLLayerID_'])


def fetch_weights(model_weights, layer_id):
    '''
    Get weights of a layer

    Parameters
    ----------
    model_weights : :class:`CASTable`
        CASTable of the model weights.
    layer_id : int
        ID of the layer.

    Returns
    -------
    list
        List of weights of the layer

    '''
    layer_weights = model_weights.query('_LayerID_ = {}'.format(layer_id))
    n = layer_weights.numrows()['numrows']
    return layer_weights.fetch(maxRows=n, to=n, sortBy='_WeightID_')['Fetch']['_Weight_'] \
                        .tolist()


def get_weights_from_dataframe(model_weights, layer_id):
    '''
    Get weights of a layer

    Parameters
    ----------
    model_weights : :class:`pandas.DataFrame`
        DataFrame of the model weights.
    layer_id : int
        ID of the layer.

    Returns
    -------
    list
        List of weights of the layer

    '''
    layer_weights = model_weights[model_weights['_LayerID_'] == layer_id]['_Weight_']
    return layer_weights.tolist()


def sas_to_onnx_activation(activation):
    ''' Convert SAS activation names to ONNX '''
    if activation.lower() == 'rectifier' or activation.lower() == 'relu':
        return 'Relu'
    elif activation.lower() == 'tanh':
        return 'Tanh'
    elif activation.lower() == 'logistic' or activation.lower() == 'sigmoid':
        return 'Log'
    elif activation.lower() == 'leaky':
        return 'LeakyRelu'
    elif activation.lower() == 'identity':
        return 'Identity'
    elif activation.lower() == 'elu':
        return 'Elu'
    elif activation.lower() == 'softplus':
        return 'Softplus'
    elif activation.lower() == 'softmax':
        return 'Softmax'
    else:
        print('WARNING: Unsupported activation: '
              + str(activation) + '. Using identity.')
        return 'Identity'


def make_onnx_activation(activation, act_input, act_output):
    ''' Make onnx activation op '''
    onnx_act = sas_to_onnx_activation(activation)
    if onnx_act == 'LeakyRelu':
        return helper.make_node(op_type=onnx_act,
                                inputs=act_input,
                                outputs=act_output,
                                alpha=0.1)
    else:
        return helper.make_node(op_type=onnx_act,
                                inputs=act_input,
                                outputs=act_output)


def get_padding(layer):
    ''' Gets the padding along each axis '''
    if layer.config.get('padding') is not None:
        return [int(layer.config['padding'])]*4
    elif (layer.config.get('padding_width') is not None and
          layer.config.get('padding_height') is None):
        return [int(layer.config['padding_width'])]*4
    elif (layer.config.get('padding_height') is not None and
          layer.config.get('padding_width') is None):
        return [int(layer.config['padding_height'])]*4
    elif (layer.config.get('padding_width') is not None and
          layer.config.get('padding_height') is not None):
        P_h = int(layer.config['padding_height'])
        P_w = int(layer.config['padding_width'])
        return [P_h, P_w]*2
    else:
        H = int(layer.config['height'])
        W = int(layer.config['width'])
        S_h, S_w = get_strides(layer)
        
        in_W = layer.src_layers[0].output_size[0]
        in_H = layer.src_layers[0].output_size[1]
        if (in_H % S_h == 0):
            pad_h = max(0, H - S_h)
        else:
            pad_h = max(0, H - (in_H % S_h))
        if (in_W % S_w == 0):
            pad_w = max(0, W - S_w)
        else:
            pad_w = max(0, W - (in_W % S_w))

        if layer.type == 'pool':
            return [0, 0, pad_h, pad_w]

        if pad_h % 2 == 0:
            P_h = P_h_ = pad_h // 2
        else:
            P_h = pad_h // 2
            P_h_ = P_h + 1
        if pad_w % 2 == 0:
            P_w = P_w_ = pad_w // 2
        else:
            P_w = pad_w // 2
            P_w_ = P_w + 1
        return [P_h, P_w, P_h_, P_w_]


def get_strides(layer):
    ''' Gets the strides along each axis '''
    if layer.config.get('stride') is not None:
        return [int(layer.config['stride'])]*2
    elif (layer.config.get('stride_horizontal') is not None and
          layer.config.get('stride_vertical') is None):
        return [int(layer.config['stride_horizontal'])]*2
    elif (layer.config.get('stride_vertical') is not None and
          layer.config.get('stride_horizontal') is None):
        return [int(layer.config['stride_vertical'])]*2
    elif (layer.config.get('stride_horizontal') is not None and
          layer.config.get('stride_vertical') is not None):
        S_h = int(layer.config['stride_vertical'])
        S_w = int(layer.config['stride_horizontal'])
        return [S_h, S_w]
    else:
        print('WARNING: Stride not specified. '
              'Setting stride to 1')
        return [1, 1]

