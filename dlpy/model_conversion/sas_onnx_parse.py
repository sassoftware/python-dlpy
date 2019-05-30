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

''' Convert ONNX models to SAS models '''

import sys
import warnings

import h5py
import numpy as np
from onnx import numpy_helper
from onnx.shape_inference import infer_shapes

from dlpy.layers import (InputLayer, Conv2d, Pooling, Dense, OutputLayer,
                         BN, Concat, Res)
from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
from dlpy.model_conversion.onnx_transforms import (ConstToInitializer,
                                                   InitReshape, InitUnsqueeze,
                                                   FuseMulAddBN)

# The supported ONNX ops that can be parsed by this module
_onnx_ops = ['Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool',
             'BatchNormalization', 'Concat', 'Gemm', 'MatMul',
             'Add', 'Sum', 'Reshape', 'Dropout', 'Flatten', 'Constant']


# mapping ONNX ops to SAS activations
_act_map = {
    'Relu': 'RELU',
    'Tanh': 'TANH',
    'LeakyRelu': 'LEAKY',
    'Log': 'LOGISTIC',
    'Softmax': 'SOFTMAX',
    'Identity': 'IDENTITY',
    'Cos': 'COS',
    'Sin': 'SIN',
    'Exp': 'EXP',
    'Elu': 'ELU',
    'Softplus': 'SOFTPLUS'
}


class OnnxParseError(ValueError):
    '''
    Used to indicate an error in parsing ONNX model definition

    '''


def onnx_to_sas(model, model_name=None, output_layer=None):
    ''' 
    Generate SAS model from ONNX model 
    
    Parameters
    ----------
    model : ONNX ModelProto
        Specifies the loaded ONNX model.
    model_name : string, optional
        Specifies the name of the model.
    output_layer : Layer object, optional
        Specifies the output layer of the model. If no output
        layer is specified, the last layer is automatically set
        as :class:`OutputLayer` with SOFTMAX activation.

    Returns
    -------
    list
        List of Layers 

    '''
    # run transforms
    graph_ = OnnxGraph.from_onnx(model.graph)
    transforms = [
        ConstToInitializer(),
        InitReshape(),
        InitUnsqueeze(),
        FuseMulAddBN()
    ]

    for transform in transforms:
        transform(graph_)
    model = graph_.make_onnx()

    # verify model ops are supported 
    for n in model.graph.node:
        if n.op_type not in _onnx_ops + list(_act_map.keys()):
            raise OnnxParseError('Unsupported op: ' + n.op_type)

    # get shapes of tensors in graph
    model = infer_shapes(model)
    graph_def = model.graph
    dlpy_layers = []

    if model_name is None:
        model_name = graph_def.name

    # nodes that correspond to sas deeplearn layers 
    sas_computation_nodes = onnx_filter_sas_layers(graph_def)

    # initializer: TensorProtos representing values to initialize
    # initialized: A list of names of the initialized tensors
    if graph_def.initializer:
        init_tensors = onnx_initializer_to_tensors(graph_def.initializer)
        initialized = [init.name for init in graph_def.initializer]
    else:
        init_tensors = []
        initialized = []

    tensor_dict = dict(init_tensors)

    # determine SAS input layers from uninitialized input ValueInfo
    uninitialized = [value_info for value_info in graph_def.input
                     if value_info.name not in initialized]

    if not uninitialized:
        raise OnnxParseError('Unable to determine input layer.')
    elif len(uninitialized) > 1:
        # TODO: support multipe input layers
        raise OnnxParseError('Unable to determine input layer.')
    else:
        input_layer = onnx_input_layer(uninitialized[0])
        dlpy_layers.append(input_layer)

    # create SAS layers from the ONNX nodes 
    for node in sas_computation_nodes:
        layer = onnx_extract_sas_layer(graph_def, node, dlpy_layers)
        dlpy_layers.append(layer)
    
    # apply activations
    for node in graph_def.node:
        if node.op_type in _act_map.keys():
            # handle output layer activations separately 
            if node.op_type == 'Softmax':
                continue
            previous = onnx_find_previous_compute_layer(graph_def, node)
            if len(previous) != 1:
                print('Warning: Unable to apply activation for node '
                      + str(node.name) + '.')
                continue
            for layer in dlpy_layers:
                # TODO: better checks for valid activations 
                if layer.name == previous[0].name: 
                    if 'act' in layer.config.keys():
                        layer.config.update(act=_act_map.get(node.op_type))
                    else:
                        print('Warning: Unable to apply activation for '
                              + layer.name + ' layer.')

    # apply dropout
    for node in graph_def.node:
        if node.op_type == 'Dropout':
            previous = onnx_find_previous_compute_layer(graph_def, node)
            if len(previous) != 1:
                print('Warning: Unable to apply dropout. '
                      'More than one source layer found.')
                continue
            for layer in dlpy_layers:
                if layer.name == previous[0].name:
                    if 'dropout' in layer.config.keys():
                        layer.config.update(dropout=node.attribute[0].f)
                    else:
                        print('Warning: Unable to apply dropout for'
                              + layer.name + ' layer')

    # write weights hdf5
    hdf5_out = write_weights_hdf5(dlpy_layers, graph_def, tensor_dict, model_name)

    # add output layer
    # if output_layer is not specified, output layer defaults to SOFTMAX
    if output_layer is None:
        # if previous layer is fc, we can replace it with output layer
        if dlpy_layers[-1].type == 'fc':
            last_layer = dlpy_layers.pop()
            out_layer = OutputLayer(name=last_layer.name,
                                    act='SOFTMAX',
                                    n=last_layer.config['n'],
                                    src_layers=last_layer.src_layers)
            dlpy_layers.append(out_layer)
        # if previous layer is not fc, default to loss layer only
        else:
            n = dlpy_layers[-1].output_size[-1]
            out_layer = OutputLayer(name='output',
                                    act='IDENTITY',
                                    n=n,
                                    include_bias=False,
                                    src_layers=[dlpy_layers[-1]])
            dlpy_layers.append(out_layer)
            identity = np.identity(n).astype(np.float32)
            f = h5py.File(hdf5_out)
            f['output/output/kernel:0'] = identity
            f.close()
    else:
        # connect output_layer to previous layer
        output_layer.src_layers = [dlpy_layers[-1]]
        if not output_layer.name:
            output_layer.name = 'output'
        dlpy_layers.append(output_layer)

    return dlpy_layers


def onnx_initializer_to_tensors(initializer):
    ''' 
    Convert ONNX graph initializer to tensors 
    
    Parameters
    ----------
    initializer : list of ONNX TensorProto
        Specifies the initializer of the graph.

    Returns
    -------
    list of numpy.ndarray

    '''

    return [(init.name,
             numpy_helper.to_array(init))
            for init in initializer]


def onnx_filter_sas_layers(graph):
    ''' 
    Filter nodes that correspond to SAS Deep learning layer types 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    
    Returns
    -------
    list of ONNX NodeProto

    '''

    sas_layers = [node for node in graph.node
                  if is_compute_layer(graph, node)]

    return sas_layers


def onnx_input_layer(value_info):
    ''' 
    Construct Input Layer 
    
    Parameters
    ----------
    value_info : ONNX ValueInfoProto
        Specifies a ValueInfoProto object.

    Returns
    -------
    :class:`InputLayer`
    
    '''
    input_layer_name = value_info.name
    _, C, H, W = list(d.dim_value for d in
                      value_info.type.tensor_type.shape.dim)
    return InputLayer(n_channels=C, width=W, height=H, name=input_layer_name)


def onnx_extract_sas_layer(graph, node, layers):
    '''
    Generate SAS DeepLearn Layer from ONNX node

    Parameters
    ----------
    graph : ONNX :class:`GraphProto`
        Specifies the ONNX graph.
    node : ONNX :class:`NodeProto`
        Specifies the ONNX node.
    layers : list of Layers
        The sequential layers of the model.

    Returns
    -------
    :class:`Layer`
        Layer object corresponding to the ONNX node.

    '''
    if node.op_type == 'Conv':
        return onnx_extract_conv(graph, node, layers)
    elif node.op_type == 'MaxPool':
        return onnx_extract_pool(graph, node, layers, pool='MAX')
    elif node.op_type == 'AveragePool':
        return onnx_extract_pool(graph, node, layers, pool='AVERAGE')
    elif node.op_type == 'BatchNormalization':
        return onnx_extract_batchnormalization(graph, node, layers)
    elif node.op_type == 'Concat':
        return onnx_extract_concat(graph, node, layers)
    elif node.op_type == 'Gemm':
        return onnx_extract_gemm(graph, node, layers)
    elif node.op_type == 'MatMul':
        return onnx_extract_matmul(graph, node, layers)
    elif node.op_type in ['Add', 'Sum']:
        return onnx_extract_residual(graph, node, layers)
    elif node.op_type == 'GlobalAveragePool':
        return onnx_extract_globalpool(graph, node, layers)
    else:
        raise OnnxParseError('Unsupported ONNX op: '
                             + str(node.name) + ', '
                             + str(node.op_type))


def find_input_layer_name(graph):
    ''' 
    Determine the name of the input layer 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies the GraphProto object.

    Returns
    -------
    string

    '''
    initialized = [init.name for init in graph.initializer]
    uninitialized = [value_info.name for value_info in graph.input
                     if value_info.name not in initialized]
    if not uninitialized:
        raise OnnxParseError('Unable to determine input layer.')
    if len(uninitialized) > 1:
        raise OnnxParseError('Unable to determine input layer.')
    return uninitialized[0]


def get_dlpy_layer(layers, name):
    ''' 
    Get a layer by name from list of layers 
    
    Parameters
    ----------
    layers : list of Layers
        Specifies a list of Layers.
    name : string
        Specifies the name of a Layer.

    Returns
    -------
    Layer, or None
        The layer matching the name, or None.
    ''' 
    for layer in layers:
        if layer.name == name:
            return layer
    return None


def onnx_extract_conv(graph, node, layers):
    ''' 
    Construct convo layer from ONNX op 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.

    Returns
    -------
    :class:`Conv2d`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]
    
    src = [get_dlpy_layer(layers, i) for i in src_names]

    height = None
    width = None
    stride = None
    stride_horizontal = None
    stride_vertical = None
    padding = None
    padding_height = None
    padding_width = None
    n_filters = None
    include_bias = False
    act = 'identity'

    # if padding is not present, default to 0
    is_padding = False

    attributes = node.attribute
    for attr in attributes:
        if attr.name == 'kernel_shape':
            height, width = attr.ints
        elif attr.name == 'strides':
            stride_vertical, stride_horizontal = attr.ints
            # only specify one of stride and stride_horizontal
            if stride_horizontal == stride_vertical:
                stride = stride_horizontal
                stride_horizontal = None
                stride_vertical = None
        elif attr.name == 'auto_pad':
            is_padding = True
            attr_s = attr.s.decode('utf8')
            if attr_s == 'SAME_UPPER' or attr_s == 'SAME_LOWER':
                continue
            elif attr_s == 'NOTSET':
                continue
            else: # 'VALID'
                padding = 0
        elif attr.name == 'pads':
            is_padding = True
            padding_height, padding_width, p_h2, p_w2 = attr.ints
            if padding_height != p_h2 or padding_width != p_w2:
                print('Warning: Unequal padding not supported for '
                      + node.name + ' setting equal padding instead.')
                padding_height = max(padding_height, p_h2)
                padding_width = max(padding_width, p_w2)

    if not is_padding:
        padding = 0

    # check if weight tensor is in initializer
    for init in graph.initializer:
        if init.name == node.input[1]:
            n_filters = numpy_helper.to_array(init).shape[0]
    
    # if not in initializer, check inferred shapes in graph
    if n_filters is None:
        for v in graph.value_info:
            if v.name == node.input[1]:
                n_filters = v.type.tensor_type.shape.dim[0].dim_value

    # check if bias is specified in conv op
    if len(node.input) == 3:
        include_bias = True
    # check if bias is added by the next op
    else:
        out = onnx_get_out_nodes(graph, node) 
        for n in out:
            if is_bias_op(graph, n):
                include_bias = True
   
    return Conv2d(n_filters=n_filters,
                  width=width,
                  height=height,
                  stride=stride,
                  name=node.name,
                  stride_horizontal=stride_horizontal,
                  stride_vertical=stride_vertical,
                  padding=padding,
                  padding_width=padding_width,
                  padding_height=padding_height,
                  act=act,
                  include_bias=include_bias,
                  src_layers=src) 

    
def onnx_extract_pool(graph, node, layers, pool='MAX'):
    ''' 
    Construct pool layer from ONNX op 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.
    pool : str, optional
        Specifies the type of pooling.
        Default: MAX

    Returns
    -------
    :class:`Pooling`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]
    
    src = [get_dlpy_layer(layers, i) for i in src_names]

    height = None
    padding = None
    padding_height = None
    padding_width = None
    stride = None
    stride_horizontal = None
    stride_vertical = None
    width = None
    
    # if padding is not present, default to 0
    is_padding = False

    for attr in node.attribute:
        if attr.name == 'kernel_shape':
            height, width = attr.ints
        elif attr.name == 'strides':
            stride_vertical, stride_horizontal = attr.ints
            # only specify one of stride and stride_horizontal
            if stride_horizontal == stride_vertical:
                stride = stride_horizontal
                stride_horizontal = None
                stride_vertical = None
        elif attr.name == 'auto_pad':
            is_padding = True
            attr_s = attr.s.decode('utf8')
            if attr_s == 'SAME_UPPER' or attr_s == 'SAME_LOWER':
                continue
            elif attr_s == 'NOTSET':
                continue
            else: # 'VALID'
                padding = 0
        elif attr.name == 'pads':
            is_padding = True
            padding_height, padding_width, p_h2, p_w2 = attr.ints
            if padding_height != p_h2 or padding_width != p_w2:
                print('WARNING: Unequal padding not supported for '
                      + node.name + ' Setting auto padding instead.')
                if padding_height == 0 and p_h2 != 0:
                    padding_height = None
                else:
                    padding_height = max(padding_height, p_h2)
                if padding_width == 0 and p_w2 != 0:
                    padding_width = None
                else:
                    padding_width = max(padding_width, p_w2)

    if not is_padding:
        padding = 0
    
    return Pooling(width=width,
                   height=height,
                   stride=stride,
                   name=node.name,
                   stride_horizontal=stride_horizontal,
                   stride_vertical=stride_vertical,
                   padding=padding,
                   padding_width=padding_width,
                   padding_height=padding_height,
                   pool=pool,
                   src_layers=src)


def onnx_extract_globalpool(graph, node, layers):
    ''' 
    Construct global pool layer from ONNX op 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.

    Returns
    -------
    :class:`Pooling`
   
    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]
    
    src = [get_dlpy_layer(layers, i) for i in src_names]
    
    # check the shape of the input to pool op
    _, C, height, width = onnx_get_shape(graph, node.input[0])
    
    return Pooling(width=width,
                   height=height,
                   stride=1,
                   name=node.name,
                   padding=0,
                   pool='AVERAGE',
                   src_layers=src)


def onnx_extract_batchnormalization(graph, node, layers):
    ''' 
    Construct batchnorm layer from ONNX op 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.
    
    Returns
    -------
    :class:`BN`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]

    src = [get_dlpy_layer(layers, i) for i in src_names]

    return BN(name=node.name,
              act='identity',
              src_layers=src)


def onnx_extract_concat(graph, node, layers):
    ''' 
    Construct concat layer from ONNX op 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.
    
    Returns
    -------
    :class:`Concat`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]

    src = [get_dlpy_layer(layers, i) for i in src_names]

    return Concat(name=node.name,
                  act='identity',
                  src_layers=src)


def onnx_extract_gemm(graph, node, layers):
    ''' 
    Construct FC layer from ONNX op 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.
   
    Returns
    -------
    :class:`Dense`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]

    src = [get_dlpy_layer(layers, i) for i in src_names]
    
    include_bias = True
    act = 'identity'
    neurons = None

    # determine dimensions of the multiply 
    a_shape = None
    b_shape = None
    # check initializer for weight tensors
    for init in graph.initializer:
        if init.name == node.input[0]:
            a_shape = numpy_helper.to_array(init).shape
        if init.name == node.input[1]:
            b_shape = numpy_helper.to_array(init).shape
    
    # check inferred shapes in graph
    for v in graph.value_info:
        if v.name == node.input[0]:
            try:
                a_shape = (v.type.tensor_type.shape.dim[0].dim_value, 
                           v.type.tensor_type.shape.dim[1].dim_value)
            except IndexError:
                pass
        if v.name == node.input[1]:
            try:
                b_shape = (v.type.tensor_type.shape.dim[0].dim_value, 
                           v.type.tensor_type.shape.dim[1].dim_value)
            except IndexError:
                pass

    if a_shape is None and b_shape is None:
        raise OnnxParseError('Unable to determine number of neurons '
                             'in FC layer.')
    elif a_shape is None or b_shape is None:
        prev_out = layers[-1].output_size
        if isinstance(prev_out, int):
            fc_input_dim = prev_out
        else:
            fc_input_dim = 1
            for d in prev_out:
                fc_input_dim *= int(d)
        if a_shape is None:
            a_shape = (1, fc_input_dim)
        else:
            b_shape = (1, fc_input_dim)
    
    # check if transpose
    for attr in node.attribute:
        if attr.name == 'transA':
            if attr.i == 1:
                a_shape = (a_shape[1], a_shape[0])
        elif attr.name == 'transB':
            if attr.i == 1:
                b_shape = (b_shape[1], b_shape[0])

    # set number of neurons according to shape
    if a_shape[0] == 1:
        neurons = b_shape[1]
    else:
        neurons = a_shape[0]
    
    return Dense(n=neurons,
                 name=node.name,
                 act=act,
                 include_bias=include_bias,
                 src_layers=src)


def onnx_extract_matmul(graph, node, layers):
    ''' 
    Construct FC layer from ONNX op 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.
    
    Returns
    -------
    :class:`Dense`

    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]

    src = [get_dlpy_layer(layers, i) for i in src_names]
    
    include_bias = False
    act = 'identity'
    neurons = None

    # determine dimensions of the multiply 
    a_shape = None
    b_shape = None
    # check initializer for weight tensors
    for init in graph.initializer:
        if init.name == node.input[0]:
            a_shape = numpy_helper.to_array(init).shape
        if init.name == node.input[1]:
            b_shape = numpy_helper.to_array(init).shape
    
    # check inferred shapes in graph
    for v in graph.value_info:
        if v.name == node.input[0]:
            a_shape = (v.type.tensor_type.shape.dim[0].dim_value, 
                       v.type.tensor_type.shape.dim[1].dim_value)
        if v.name == node.input[1]:
            b_shape = (v.type.tensor_type.shape.dim[0].dim_value, 
                       v.type.tensor_type.shape.dim[1].dim_value)

    if a_shape is None or b_shape is None:
        raise OnnxParseError('Unable to determine number of neurons '
                             'in FC layer.')
    
    # set number of neurons according to shape
    if a_shape[0] == 1:
        neurons = b_shape[1]
    else:
        neurons = a_shape[0]
    
    # check if bias is added by the next op
    out = onnx_get_out_nodes(graph, node) 
    for n in out:
        if is_bias_op(graph, n):
            include_bias = True

    return Dense(n=neurons,
                 name=node.name,
                 act=act,
                 include_bias=include_bias,
                 src_layers=src)


def onnx_extract_residual(graph, node, layers):
    ''' 
    Construct residual layer from ONNX op 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    layers : list of Layers
        Specifies the existing layers of a model.

    Returns
    -------
    :class:`Res`
   
    '''
    previous = onnx_find_previous_compute_layer(graph, node)
    
    if not previous:
        src_names = [find_input_layer_name(graph)]
    else:
        src_names = [p.name for p in previous]

    src = [get_dlpy_layer(layers, i) for i in src_names]

    return Res(name=node.name,
               act='identity',
               src_layers=src)


def onnx_get_node(graph, name):
    ''' 
    Get ONNX node from graph by name 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.
    
    Returns
    -------
    NodeProto

    '''
    for node in graph.node:
        if node.name == name:
            return node


def onnx_get_input_nodes(graph, node):
    ''' 
    Return all nodes that are inputs for a node 

    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    list
        list of NodeProto
    
    '''
    in_nodes = []
    for i in node.input:
        for n in graph.node:
            if i in n.output:
                in_nodes.append(n)
    return in_nodes


def onnx_get_out_nodes(graph, node):
    ''' 
    Return all nodes that connect to output 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    list
        list of NodeProto

    '''
    out_nodes = []
    for i in node.output:
        for n in graph.node:
            if i in n.input:
                out_nodes.append(n)
    return out_nodes


def onnx_find_previous_compute_layer(graph, node):
    ''' 
    Determine a node's previous corresponding SAS compute layer 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    list
        List of NodeProto 

    '''
    src = []

    def f(graph, node, src_layers):
        in_nodes = onnx_get_input_nodes(graph, node)
        for n in in_nodes:
            if is_compute_layer(graph, n):
                src_layers.append(n)
            else:
                f(graph, n, src_layers) 

    f(graph, node, src)
    return src

        
def is_compute_layer(graph, node):
    ''' 
    Determine if this ONNX node corresponds to a SAS layer 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    bool

    '''
    # 'Add' and 'Sum' are handled separately since they may be
    # either a bias op for previous layer, or a SAS residual layer
    # TODO: add reshape 
    sas_layers = ['Conv', 'MaxPool', 'AveragePool', 'GlobalAveragePool',
                  'BatchNormalization', 'Concat', 'Gemm', 'MatMul']

    if node.op_type in ['Add', 'Sum']:
        if is_residual_layer(graph, node):
            return True
    else:
        if node.op_type in sas_layers:
            return True

    return False


def is_residual_layer(graph, node):
    ''' 
    Correctly identify add op as residual layer 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    bool
    
    '''
    # or add/sum for residual layer
    if node.op_type not in ['Add', 'Sum']:
        return False

    # check that an input to add op is not initialized
    # i.e. a bias term
    initialized = [init.name for init in graph.initializer]
    
    # if initializer->reshape->input, return False
    for n in graph.node:
        if n.op_type != 'Reshape':
            continue
        a, b = n.input
        if a in initialized and b in initialized:
            if n.output in node.input:
                return False

    # if an input comes from initializer, return False
    for i in node.input:
        if i in initialized:
            return False

    return True


def is_bias_op(graph, node):
    ''' 
    Correctly identify bias op 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    bool

    ''' 
    if node.op_type not in ['Add', 'Sum']:
        return False
    
    initialized = [init.name for init in graph.initializer]

    # if an input comes from initializer, return True 
    for i in node.input:
        if i in initialized:
            return True

    # if initializer->reshape->input, return True 
    for n in graph.node:
        if n.op_type != 'Reshape':
            continue
        a, b = n.input
        if a in initialized and b in initialized:
            if n.output in node.input:
                return True

    return False


def onnx_find_next_activation(graph, node):
    ''' 
    Check if an activation follows current compute node 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    node : ONNX NodeProto
        Specifies a NodeProto object.

    Returns
    -------
    string
        Name of the ONNX activation op

    '''
    # if so, return that. Otherwise, return None
    # TODO: use this to find activations during layer generation
    activation_ops = ['Relu', 'Tanh', 'LeakyRelu', 'Log', 'Identity']
    
    out = onnx_get_out_nodes(graph, node)
     
    if len(out) != 1:
        return None
    else:
        if is_compute_layer(graph, out[0]):
            return None
        elif out[0].op_type in activation_ops:
            return out[0]
        else:
            return onnx_find_next_activation(graph, out[0])


def onnx_get_shape(graph, tensor):
    ''' 
    Get shape from valueinfo 
    
    Parameters
    ----------
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    tensor : ONNX ValueInfoProto 
        Specifies a ValueInfoProto object.

    Returns
    -------
    list
        list of ints of the dimensions

    '''
    for i in graph.value_info:
        if i.name == tensor:
            return [d.dim_value for d in i.type.tensor_type.shape.dim]


def write_weights_hdf5(layers, graph, tensor_dict, name):
    ''' 
    Write SAS compatible HDF5 weights file 
    
    Parameters
    ----------
    layers : list of Layers
        Specifies the layers of the model.
    graph : ONNX GraphProto
        Specifies a GraphProto object.
    tensor_dict : dict of numpy.ndarray
        Specifies the dictionary of weight tensors.
    name : string
        Specifies the name of the model.

    '''
    import os
    temp_HDF5 = os.path.join(os.getcwd(), '{}_weights.onnxmodel.h5'.format(name))
    f_out = h5py.File(temp_HDF5, 'w')
    weight_layers = [l for l in layers if l.type in ['convo', 'fc', 'batchnorm']]
    f_out.attrs['layer_names'] = [l.name.encode('utf8') for l in weight_layers]
    for layer in weight_layers:
        new_weight_names = []
        g_out = f_out.create_group(layer.name)
        node = onnx_get_node(graph, layer.name)
        weights = [np.array(tensor_dict[i], dtype=np.float32) for i in node.input
                                  if tensor_dict.get(i) is not None]

        if layer.type in ['convo', 'fc']:
            # check bias op following the node
            # to see if we need to include any bias weights
            for n in onnx_get_out_nodes(graph, node):
                if is_bias_op(graph, n):
                    for i in n.input:
                        if tensor_dict.get(i) is not None:
                            weights.append(tensor_dict[i].flatten())
            for w in weights:
                if len(w.shape) > 1:
                    dset_name = layer.name + '/' + 'kernel:0'
                    # check if need to transpose fc weight
                    if len(w.shape) == 2:
                        # check if transposed was specified in Gemm op
                        if node.op_type == 'Gemm':
                            for attr in node.attribute:
                                if attr.name == 'transB':
                                    if attr.i == 1:
                                        w = np.transpose(w, (1, 0))
                        if w.shape[1] == layer.config['n']:
                            w = np.transpose(w, (1,0))
                    g_out.create_dataset(dset_name.encode('utf8'), data=w)
                    new_weight_names.append(dset_name.encode('utf8'))
                else:
                    dset_name = layer.name + '/' + 'bias:0'
                    g_out.create_dataset(dset_name.encode('utf8'), data=w)
                    new_weight_names.append(dset_name.encode('utf8'))
        elif layer.type == 'batchnorm':
            template_names = ['gamma:0', 'beta:0', 'moving_mean:0', 
                              'moving_variance:0']
            template_names = [layer.name + '/' + i for i in template_names]
            if len(weights) != 4:
                raise OnnxParseError('Incorrect batchnorm weights') 
            for idx, w in enumerate(weights):
                g_out.create_dataset(template_names[idx].encode('utf8'), data=w) 
                new_weight_names.append(template_names[idx].encode('utf8'))

        g_out.attrs['weight_names'] = new_weight_names

    f_out.close()
    print('NOTE: Successfully written weights file as '
          + temp_HDF5)
    return temp_HDF5

