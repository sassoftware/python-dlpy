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

import os
import types
try:
    import caffe
    import caffe.draw
    from caffe.proto import caffe_pb2
    from caffe.pycaffe import *
except:
    raise ImportError('the following required module(s) is(are) not found: caffe')

from google.protobuf import text_format
from .write_caffe_model_parm import *
from .write_sas_code import *

caffe_activation_types = ['relu', 'prelu', 'elu', 'sigmoid', 'tanh', 'softmax', 'softmaxwithloss']
common_layers = ['data', 'memorydata', 'convolution', 'batchnorm', 'pooling', 'innerproduct', 'eltwise']


class CaffeParseError(ValueError):
    '''
    Used to indicate an error in parsing Caffe model definition
    '''


def caffe_to_sas(network_file, sas_file, model_name, network_param=None, sas_hdf5=None, phase=caffe.TEST,
                 verbose=False):
    '''
    Function to generate a SAS deep learning model from Caffe definition

    Parameters:

    ----------
    network_file : [string]
       Fully qualified file name of network definition file (*.prototxt)
    sas_file : [string]
       Fully qualified file name of SAS deep learning Python model definition
    model_name : [string]
       Name for deep learning model
    network_param : [string]
       Fully qualified file name of network parameter file (*.caffemodel)
    sas_hdf5 : [string]
       Fully qualified file name of SAS-compatible network parameter file (*.caffemodel.h5)
    phase : [int]
       One of {caffe.TRAIN, caffe.TEST, None}.
    verbose : [bool]
       To view all Caffe information messages, set to True.

    Returns
    -------

    '''

    # open output file
    try:
        fout = open(sas_file, "w")
    except IOError:
        sys.exit("Unable to create file " + sas_file)

    try:

        # initialize Caffe logging facility
        caffe.init_log(0, verbose)

        # instantiate a model and read network parameters
        if (network_param is None):
            model = caffe.Net(network_file, phase)
        else:
            model = caffe.Net(network_file, phase, weights=network_param)
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(network_file + ".tmp").read(), net)

        # remove temporary file created
        if os.path.isfile(network_file + ".tmp"):
            os.remove(network_file + ".tmp")

        # identify common Caffe/SAS computation layers
        layer_list = []
        for layer in net.layer:
            include_layer = False
            if (len(layer.include) == 0):
                include_layer = True
            else:
                for layer_phase in layer.include:
                    if (caffe.TEST == layer_phase.phase):
                        include_layer = True

            # exclude layers not implemented (or implemented in a different fashion)
            if (layer.type.lower() not in common_layers):
                include_layer = False

            if include_layer:
                layer_list.append(make_composite_layer(layer))

        # associate activations with computation layers
        for layer in net.layer:
            layer_type = layer.type.lower()
            if (layer_type in ['relu', 'prelu', 'elu', 'sigmoid', 'tanh']):
                layer_index = None
                for ii in range(len(layer_list)):
                    if (layer.top[0] == layer_list[ii].layer_parm.top[0]):
                        layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        "ERROR: activation layer " + layer.name + " is not associated with any computation layer.")

        # associate dropout with computation layers
        for layer in net.layer:
            layer_type = layer.type.lower()
            if (layer_type == 'dropout'):
                layer_index = None
                for ii in range(len(layer_list)):
                    if (layer.top[0] == layer_list[ii].layer_parm.top[0]):
                        layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        "ERROR: dropout layer " + layer.name + " is not associated with any computation layer.")

        # associate softmax with a fully-connected layer
        for layer in net.layer:
            layer_type = layer.type.lower()
            if (layer_type in ['softmax', 'softmaxwithloss']):
                layer_index = None
                for ii in range(len(layer_list)):
                    for jj in range(len(layer.bottom)):
                        if (layer.bottom[jj] == layer_list[ii].layer_parm.top[0]):
                            layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        "ERROR: softmax layer " + layer.name + " is not associated with any fully-connected layer.")

        # determine source layer(s) for computation layers
        for ii in range(len(layer_list)):
            for kk in range(len(layer_list[ii].layer_parm.bottom)):
                name = None
                for jj in range(ii):
                    if (layer_list[ii].layer_parm.bottom[kk] == layer_list[jj].layer_parm.top[0]):
                        name = layer_list[jj].layer_parm.name

                if name:
                    layer_list[ii].source_layer.append(name)

        # associate scale layer with batchnorm layer
        for layer in net.layer:
            if (layer.type.lower() == 'scale'):
                bn_found = False
                for ii in range(len(layer_list)):
                    if ((layer_list[ii].layer_parm.type.lower() == 'batchnorm') and
                            (layer_list[ii].layer_parm.top[0] == layer.top[0])):
                        layer_list[ii].related_layers.append(layer)
                        bn_found = True
                        break

                if not bn_found:
                    raise CaffeParseError(
                        "ERROR: scale layer " + layer.name + " is not associated with a batch normalization layer")

        # loop over included layers
        for clayer in layer_list:
            layer_type = clayer.layer_parm.type.lower()
            if (layer_type == 'pooling'):  # average/max pooling
                sas_code = caffe_pooling_layer(clayer, model_name)
            elif (layer_type == 'convolution'):  # 2D convolution
                sas_code = caffe_convolution_layer(clayer, model_name)
            elif (layer_type == 'batchnorm'):  # batch normalization
                sas_code = caffe_batch_normalization_layer(clayer, model_name)
            elif (layer_type in ['data', 'memorydata']):  # input layer
                sas_code = caffe_input_layer(clayer, model_name)
            elif (layer_type == 'eltwise'):  # residual
                sas_code = caffe_residual_layer(clayer, model_name)
            elif (layer_type == 'innerproduct'):  # fully connected
                sas_code = caffe_full_connect_layer(clayer, model_name)
            else:
                raise CaffeParseError("ERROR: " + layer_type + " is an unsupported layer type")

            # write SAS code associated with Caffe layer
            if sas_code:
                fout.write(sas_code + "\n\n")
            else:
                raise CaffeParseError("ERROR: unable to generate SAS definition for layer " + clayer.layer_parm.name)

    except CaffeParseError as err_msg:
        print(err_msg)
    finally:
        sas_code = write_main_entry(model_name)
        fout.write(sas_code + "\n")
        fout.close()

    # convert from BINARYPROTO to HDF5
    if (sas_hdf5 is not None):
        write_caffe_hdf5(model, layer_list, sas_hdf5)


# parse parameters for pooling layer and generate equivalent SAS code
def caffe_pooling_layer(clayer, model_name):
    '''
    Function to extract pooling layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning pooling layer definition
    '''

    layer_parm = clayer.layer_parm

    # list defining PoolingParameter data structure --> keep in sync with caffe.proto
    dstruct = [{"field": "pool", "repeated": False},
               {"field": "pad", "repeated": False},
               {"field": "pad_h", "repeated": False},
               {"field": "pad_w", "repeated": False},
               {"field": "kernel_size", "repeated": False},
               {"field": "kernel_h", "repeated": False},
               {"field": "kernel_w", "repeated": False},
               {"field": "stride", "repeated": False},
               {"field": "stride_h", "repeated": False},
               {"field": "stride_w", "repeated": False},
               {"field": "engine", "repeated": False},
               {"field": "global_pooling", "repeated": False}]

    # read pooling parameters
    pooling_param = getattr(layer_parm, 'pooling_param', None)
    if (pooling_param is not None):
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = dstruct[ii]['field'] + "=extract_repeated_attr" \
                           + "(pooling_param,'" \
                           + dstruct[ii]['field'] + "')"
            else:
                code_str = dstruct[ii]['field'] + "=extract_attr" \
                           + "(pooling_param,'" \
                           + dstruct[ii]['field'] + "')"

            exec(code_str)
    else:
        raise CaffeParseError("ERROR: no pooling parameters given")

    # define parameters needed by SAS pooling layer

    # pooling type
    if (pool == 0):
        pool_type = 'max'
    elif (pool == 1):
        pool_type = 'mean'
    else:
        CaffeParseError("ERROR: invalid pooling type specified for layer = " + layer_parm.name)

    # stride (vertical)
    if (stride_h is not None) and (stride_h > 0):
        tmp_stride_h = stride_h
    else:
        if (stride is None) or (stride == 0):
            tmp_stride_h = 1
        else:
            tmp_stride_h = stride

    # stride (horizontal)
    if (stride_w is not None) and (stride_w > 0):
        tmp_stride_w = stride_w
    else:
        if (stride is None) or (stride == 0):
            tmp_stride_w = 1
        else:
            tmp_stride_w = stride

    # horizontal/vertical stride must agree
    if (tmp_stride_w != tmp_stride_h):
        CaffeParseError("ERROR: horizontal/vertical strides do not agree for layer = " + layer_parm.name)
    else:
        common_stride = tmp_stride_w

    # height of kernel
    if (kernel_h is not None) and (kernel_h > 0):
        height = kernel_h
    else:
        if (kernel_size is None):
            CaffeParseError("ERROR: unable to set kernel height for layer = " + layer_parm.name)
        else:
            height = kernel_size

    # width of kernel
    if (kernel_w is not None) and (kernel_w > 0):
        width = kernel_w
    else:
        if (kernel_size is None):
            CaffeParseError("ERROR: unable to set kernel width for layer = " + layer_parm.name)
        else:
            width = kernel_size

    # determine dropout
    dropout = extract_dropout(clayer)
    if (dropout is None):
        dropout = 0

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers != 1):
        raise CaffeParseError("ERROR: pooling layer requires one input layer, " + str(num_layers) + " provided")

    return write_pooling_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                               width=str(width), height=str(height), stride=str(common_stride),
                               type=pool_type, dropout=str(dropout), src_layer=source_layer)


# parse parameters for convolution layer and generate equivalent SAS code
def caffe_convolution_layer(clayer, model_name):
    '''
    Function to extract convolution layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning convolution layer definition
    '''

    layer_parm = clayer.layer_parm

    # list defining ConvolutionParameter data structure --> keep in sync with caffe.proto
    dstruct = [{"field": "num_output", "repeated": False},
               {"field": "bias_term", "repeated": False},
               {"field": "pad", "repeated": True},
               {"field": "kernel_size", "repeated": True},
               {"field": "stride", "repeated": True},
               {"field": "dilation", "repeated": True},
               {"field": "pad_h", "repeated": False},
               {"field": "pad_w", "repeated": False},
               {"field": "kernel_h", "repeated": False},
               {"field": "kernel_w", "repeated": False},
               {"field": "stride_h", "repeated": False},
               {"field": "stride_w", "repeated": False},
               {"field": "group", "repeated": False},
               {"field": "weight_filler", "repeated": False},
               {"field": "bias_filler", "repeated": False},
               {"field": "engine", "repeated": False},
               {"field": "axis", "repeated": False},
               {"field": "force_nd_im2col", "repeated": False}]

    # read convolution parameters
    convolution_param = getattr(layer_parm, 'convolution_param', None)
    if (convolution_param is not None):
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = dstruct[ii]['field'] + "=extract_repeated_attr" \
                           + "(convolution_param,'" \
                           + dstruct[ii]['field'] + "')"
            else:
                code_str = dstruct[ii]['field'] + "=extract_attr" \
                           + "(convolution_param,'" \
                           + dstruct[ii]['field'] + "')"

            exec(code_str)
    else:
        raise CaffeParseError("ERROR: no convolution parameters given")

    # define parameters needed by SAS convolution layer
    # bias
    if (bias_term is not None):
        if bias_term:
            nobias = 'False'
        else:
            nobias = 'True'
    else:
        nobias = 'False'

    # number of output layers
    if (num_output is None):
        CaffeParseError("ERROR: num_output not provided for layer = " + layer_parm.name)

    # stride (vertical)
    if (stride_h is not None) and (stride_h > 0):
        tmp_stride_h = stride_h
    else:
        if (stride is None) or (stride == 0):
            tmp_stride_h = 1
        else:
            tmp_stride_h = stride

    # stride (horizontal)
    if (stride_w is not None) and (stride_w > 0):
        tmp_stride_w = stride_w
    else:
        if (stride is None) or (stride == 0):
            tmp_stride_w = 1
        else:
            tmp_stride_w = stride

    # horizontal/vertical stride must agree
    if (tmp_stride_w != tmp_stride_h):
        CaffeParseError("ERROR: horizontal/vertical strides do not agree for layer = " + layer_parm.name)
    else:
        common_stride = tmp_stride_w

    # height of kernel
    if (kernel_h is not None) and (kernel_h > 0):
        height = kernel_h
    else:
        if (kernel_size is None):
            CaffeParseError("ERROR: unable to set kernel height for layer = " + layer_parm.name)
        else:
            height = kernel_size

    # width of kernel
    if (kernel_w is not None) and (kernel_w > 0):
        width = kernel_w
    else:
        if (kernel_size is None):
            CaffeParseError("ERROR: unable to set kernel width for layer = " + layer_parm.name)
        else:
            width = kernel_size

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers != 1):
        raise CaffeParseError("ERROR: convolution layer requires one input layer, " + str(num_layers) + " provided")

    # determine activation
    act = extract_activation(clayer, 'convolution')

    # determine dropout
    dropout = extract_dropout(clayer)
    if (dropout is None):
        dropout = 0

    return write_convolution_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                                   nfilters=str(num_output), width=str(width), height=str(height),
                                   stride=str(common_stride), nobias=nobias, activation=act,
                                   dropout=str(dropout), src_layer=source_layer)


# parse parameters for batch normalization layer and generate equivalent SAS code
def caffe_batch_normalization_layer(clayer, model_name):
    '''
    Function to extract batch normalization layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning batch normalization layer definition
    '''

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers != 1):
        raise CaffeParseError(
            "ERROR: batch normalization layer requires one input layer, " + str(num_layers) + " provided")

    # determine activation
    act = extract_activation(clayer, 'batchnorm')

    return write_batch_norm_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                                  activation=act, src_layer=source_layer)


# parse parameters for input layer and generate equivalent SAS code
def caffe_input_layer(clayer, model_name):
    '''
    Function to extract input layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning input layer definition
    '''

    layer_parm = clayer.layer_parm

    # read scaling parameter
    transform_param = getattr(layer_parm, 'transform_param', None)
    if (transform_param is not None):
        scale = getattr(transform_param, 'scale', 1.0)

    # read image format parameters
    memory_data_param = getattr(layer_parm, 'memory_data_param', None)
    if (memory_data_param is not None):
        channels = getattr(memory_data_param, 'channels', -1)
        height = getattr(memory_data_param, 'height', -1)
        width = getattr(memory_data_param, 'width', -1)
    else:
        channels = -1
        height = -1
        width = -1
        print("WARNING: unable to provide parameters for image data format")

    return write_input_layer(model_name=model_name, layer_name=layer_parm.name,
                             channels=str(channels), width=str(width),
                             height=str(height), scale=str(scale))


# parse parameters for residual layer and generate equivalent SAS code
def caffe_residual_layer(clayer, model_name):
    '''
    Function to extract residual layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning residual layer definition
    '''

    layer_parm = clayer.layer_parm

    # list defining EltwiseParameter data structure --> keep in sync with caffe.proto
    dstruct = [{"field": "operation", "repeated": False},
               {"field": "coeff", "repeated": True},
               {"field": "stable_product_grad", "repeated": False}]

    # read eltwise parameters
    eltwise_param = getattr(layer_parm, 'eltwise_param', None)
    if (eltwise_param is not None):
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = dstruct[ii]['field'] + "=extract_repeated_attr" \
                           + "(eltwise_param,'" \
                           + dstruct[ii]['field'] + "')"
            else:
                code_str = dstruct[ii]['field'] + "=extract_attr" \
                           + "(eltwise_param,'" \
                           + dstruct[ii]['field'] + "')"

            exec(code_str)
    else:
        raise CaffeParseError("ERROR: no eltwise parameters given")

    # determine whether operation specified is valid
    if (operation != 1):
        raise CaffeParseError("ERROR: element-wise operation not supported")

    # determine activation
    act = extract_activation(clayer, 'residual')

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers < 2):
        raise CaffeParseError(
            "ERROR: residual layer requires two or more input layers, " + str(num_layers) + " provided")

    return write_residual_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                                activation=act, src_layer=source_layer)


# parse parameters for fully connected layer and generate equivalent SAS code
def caffe_full_connect_layer(clayer, model_name):
    '''
    Function to extract fully-connected layer parameters from LayerParameter object

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    model_name : [string]
       Deep learning model name

    Returns
    -------
    String value with SAS deep learning fully-connected layer definition
    '''

    layer_parm = clayer.layer_parm

    # list defining InnerProductParameter data structure --> keep in sync with caffe.proto
    dstruct = [{"field": "num_output", "repeated": False},
               {"field": "bias_term", "repeated": False},
               {"field": "weight_filler", "repeated": False},
               {"field": "bias_filler", "repeated": False},
               {"field": "axis", "repeated": False},
               {"field": "transpose", "repeated": False}]

    # read inner product parameters
    inner_product_param = getattr(layer_parm, 'inner_product_param', None)
    if (inner_product_param is not None):
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = dstruct[ii]['field'] + "=extract_repeated_attr" \
                           + "(inner_product_param,'" \
                           + dstruct[ii]['field'] + "')"
            else:
                code_str = dstruct[ii]['field'] + "=extract_attr" \
                           + "(inner_product_param,'" \
                           + dstruct[ii]['field'] + "')"

            exec(code_str)
    else:
        raise CaffeParseError("ERROR: no inner_product parameters given")

    # define parameters needed by SAS fully-connected layer

    # bias
    if (bias_term is not None):
        if bias_term:
            nobias = 'False'
        else:
            nobias = 'True'
    else:
        nobias = 'False'

    # number of output neurons
    if (num_output is not None):
        num_neurons = num_output
    else:
        raise CaffeParseError("ERROR: number of output neurons not specified for layer = , " + layer_parm.name)

    # check axis setting
    if (axis is not None) and (axis != 1):
        raise CaffeParseError("ERROR: axis = , " + str(axis) + " is not supported")

    # check transpose setting
    if (transpose is not None) and (transpose != False):
        raise CaffeParseError("ERROR: transpose = , " + str(transpose) + " is not supported")

    # determine activation
    act = extract_activation(clayer, 'innerproduct')

    # determine layer type
    if (act == 'softmax'):
        fc_type = 'output'
    else:
        fc_type = 'fullconnect'

    # determine dropout
    dropout = extract_dropout(clayer)
    if (dropout is None):
        dropout = 0

        # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers != 1):
        raise CaffeParseError("ERROR: fully connected layer requires one input layer, " + str(num_layers) + " provided")

    return write_full_connect_layer(model_name=model_name, layer_name=layer_parm.name,
                                    nrof_neurons=str(num_neurons), nobias=nobias,
                                    activation=act, type=fc_type, dropout=str(dropout), src_layer=source_layer)


class CompositeLayer():
    '''
    Class defining a "composite" layer object.  A composite layer
    is one that consists of common SAS/Caffe computation layers
    along with Caffe layers that share the same top blob as the
    computation layer.
    '''

    def __init__(self, layer_parm):
        self.source_layer = []
        self.layer_parm = layer_parm
        self.related_layers = []


def make_composite_layer(layer_parm):
    '''
    Function to generate a CompositeLayer object

    Parameters:

    ----------
    layer_parm :
       A Python LayerParameter object (mirrors Google protobuf definition).

    Returns
    -------
    A CompositeLayer object.

    '''
    composite_layer = CompositeLayer(layer_parm)
    return composite_layer


# map Caffe activation layer types to SAS activation types
def map_caffe_activation(layer_name, layer_type, act_type):
    '''
    Function to map Caffe activation function(s) to SAS
    activation function(s)

    Parameters:

    ----------
    layer_name : [string]
       Layer name
    layer_type : [string]
       Caffe layer type
    act_type : [string]
       Caffe activation type

    Returns
    -------
    SAS activation type
    '''

    # convolution layer
    if (layer_type in ['convolution', 'batchnorm', 'residual']):
        map_dict = {"elu": "elu", "relu": "relu", "tanh": "tanh", "sigmoid": "sigmoid"}
    elif (layer_type == 'innerproduct'):
        map_dict = {"softmax": "softmax", "elu": "elu", "relu": "relu",
                    "tanh": "tanh", "sigmoid": "sigmoid", "softmaxwithloss": "softmax"}
    else:
        raise CaffeParseError("SAS does not support activation functions for layer " + layer_name)

    if (act_type in map_dict.keys()):
        act_func = map_dict[act_type]
        if act_func is None:
            raise CaffeParseError("Activation function " + act_type + " not supported")
    else:
        raise CaffeParseError("Unknown Caffe activation function = " + act_type)

    return act_func


# extract activation from layer definition
def extract_activation(clayer, layer_type):
    '''
    Function to extract Caffe activation function from
    Caffe layer(s) sharing a common top blob

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters
    layer_type : [string]
       Caffe layer type

    Returns
    -------
    SAS activation function [default = identity]
    '''

    act = None
    if (len(clayer.related_layers) > 0):
        for ii in range(len(clayer.related_layers)):
            act_type = clayer.related_layers[ii].type.lower()
            if (act_type in caffe_activation_types):
                if (act is None):
                    act = map_caffe_activation(clayer.layer_parm.name, layer_type, act_type)
                else:
                    raise CaffeParseError("More than one activation associated with layer = " + clayer.layer_parm.name)

    if act is None:
        act = 'identity'

    return act


# extract dropout parameter
def extract_dropout(clayer):
    '''
    Function to extract dropout parameter
    from Caffe dropout layer

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters

    Returns
    -------
    Caffe dropout parameter [default = 0.0]
    '''

    dropout = None
    if (len(clayer.related_layers) > 0):
        for ii in range(len(clayer.related_layers)):
            layer_type = clayer.related_layers[ii].type.lower()
            if (layer_type == 'dropout'):
                if (dropout is None):
                    # read dropout parameters --> only one variable in DropoutParameter message,
                    # so no intervening layer_param wrapper
                    dropout_param = getattr(clayer.related_layers[ii], 'dropout_param', None)
                    if (dropout_param is not None):
                        # dropout ratio
                        dropout_ratio = getattr(dropout_param, 'dropout_ratio', 0.0)
                    else:
                        raise CaffeParseError("ERROR: no dropout parameters given")
                else:
                    raise CaffeParseError(
                        "ERROR: More than one dropout layer associated with layer = " + clayer.related_layers[
                            ii].layer_parm.name)

    return dropout


# determine source layer(s) for a given computation layer
def extract_source_layers(clayer):
    '''
    Function to construct a string representation
    of a Python list containing the layer name(s)
    of all the source layers for the current layer

    Parameters:

    ----------
    clayer : [CompositeLayer object]
       Layer parameters

    Returns
    -------
    String representation of Python list
    '''

    source_layer = "['"
    num_layers = len(clayer.source_layer)
    for ii in range(num_layers):
        source_layer = source_layer + clayer.source_layer[ii] + "','"
    source_layer = source_layer[:-2] + "]"

    return source_layer, num_layers


# extract value from repeated container object (only first returned)
def extract_repeated_attr(param, field):
    '''
    Function to extract a particular field
    defined as a RepeatedContainer object

    Parameters:

    ----------
    param : [parameter object]
       Various parameter objects defined by Google messages
    field : [string]
       Parameter field

    Returns
    -------
    Field value or None if parameter or field doesn't exist

    NOTE: only the first value is returned
    '''

    tmpval = getattr(param, field, None)
    if (tmpval is not None):
        if isinstance(tmpval, (float, int, bool)):
            y = tmpval
        elif (len(tmpval) > 0):
            y = tmpval[0]
        else:
            y = None
        return y
    else:
        return None


# extract value
def extract_attr(param, field):
    '''
    Function to extract a particular field
    from a parameter object

    Parameters:

    ----------
    param : [parameter object]
       Various parameter objects defined by Google messages
    field : [string]
       Parameter field

    Returns
    -------
    Field value or None if parameter or field doesn't exist
    '''

    tmpval = getattr(param, field, None)
    if (tmpval is not None):
        return tmpval
    else:
        return None


#########################################################################################
if __name__ == "__main__":

    # check that environment variables set
    if ('CAFFE_HDF5_PATH' not in os.environ.keys()):
        err_msg = ("Environment variable CAFFE_HDF5_PATH not set.  Please set this variable to \n"
                   "point to the directory where HDF5 files are or will be stored \n")
        sys.exit(err_msg)

    if ('CAFFE_PYTHON_PATH' in os.environ.keys()):
        sys.path.append(os.environ['CAFFE_PYTHON_PATH'])
    else:
        err_msg = ("Environment variable CAFFE_PYTHON_PATH not set.  Please set this \n"
                   "variable to point to the PyCaffe directory \n")
        sys.exit(err_msg)

    if ('CAFFE_APPLICATION_PATH' in os.environ.keys()):
        sys.path.append(os.environ['CAFFE_APPLICATION_PATH'])
    else:
        err_msg = ("Environment variable CAFFE_APPLICATION_PATH not set.  Please set this \n"
                   "variable to point to the directory where model definition files are or \n"
                   "will be stored \n")
        sys.exit(err_msg)

    model_name = "VGG19"
    if (model_name == "ResNet50"):
        # ResNet-50
        network_file = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "ResNet-50-deploy.prototxt")
        network_param = os.path.join(os.environ['CAFFE_HDF5_PATH'], "ResNet-50-model.caffemodel")
    elif (model_name == "ResNet101"):
        # ResNet-101
        network_file = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "ResNet-101-deploy.prototxt")
        network_param = os.path.join(os.environ['CAFFE_HDF5_PATH'], "ResNet-101-model.caffemodel")
    elif (model_name == "ResNet152"):
        # ResNet-152
        network_file = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "ResNet-152-deploy.prototxt")
        network_param = os.path.join(os.environ['CAFFE_HDF5_PATH'], "ResNet-152-model.caffemodel")
    elif (model_name == "LeNet"):
        # LeNet
        sys.exit('Unable to convert LeNet')
    elif (model_name == "VGG16"):
        # VGG-16
        network_file = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "VGG_ILSVRC_16_layers.prototxt")
        network_param = os.path.join(os.environ['CAFFE_HDF5_PATH'], "VGG_ILSVRC_16_layers.caffemodel")
    elif (model_name == "VGG19"):
        # VGG-19
        network_file = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "VGG_ILSVRC_19_layers_deploy.prototxt")
        network_param = os.path.join(os.environ['CAFFE_HDF5_PATH'], "VGG_ILSVRC_19_layers.caffemodel")
    else:
        sys.exit("ERROR: Unknown model specified")

    sas_python = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "sas_model.py")
    sas_hdf5 = os.path.join(os.environ['CAFFE_APPLICATION_PATH'], "sas_model.caffemodel.h5")

    caffe_to_sas(network_file, sas_python, model_name, network_param, sas_hdf5, caffe.TEST, False)
