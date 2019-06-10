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

''' Convert caffe model to sas models '''

import os

import caffe
import caffe.draw
from caffe.proto import caffe_pb2
from caffe.pycaffe import *
from google.protobuf import text_format

from .write_caffe_model_parm import write_caffe_hdf5
from .write_sas_code import (write_input_layer, write_convolution_layer,
                             write_batch_norm_layer, write_pooling_layer,
                             write_residual_layer, write_full_connect_layer,
                             write_main_entry)

caffe_activation_types = ['relu', 'prelu', 'elu', 'sigmoid', 'tanh',
                          'softmax', 'softmaxwithloss']
common_layers = ['data', 'memorydata', 'convolution', 'batchnorm',
                 'pooling', 'innerproduct', 'eltwise']


class CaffeParseError(ValueError):
    '''
    Used to indicate an error in parsing Caffe model definition

    '''


def caffe_to_sas(network_file, model_name, network_param=None,
                 phase=caffe.TEST, verbose=False):
    '''
    Generate a SAS deep learning model from Caffe definition

    Parameters
    ----------
    network_file : string
       Fully qualified file name of network definition file (*.prototxt).
    sas_file : string
       Fully qualified file name of SAS deep learning Python model definition.
    model_name : string
       Name for deep learning model.
    network_param : string, optional
       Fully qualified file name of network parameter file (*.caffemodel).
    phase : int, optional
       One of {caffe.TRAIN, caffe.TEST, None}.
    verbose : bool, optional
       To view all Caffe information messages, set to True.

    '''

    # open output file

    try:
        output_code = ''
        # initialize Caffe logging facility
        caffe.init_log(0, verbose)

        # instantiate a model and read network parameters
        if (network_param is None):
            model = caffe.Net(network_file, phase)
        else:
            model = caffe.Net(network_file, phase, weights=network_param)
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(network_file).read(), net)

        # identify common Caffe/SAS computation layers
        layer_list = []
        for layer in net.layer:
            include_layer = False
            if len(layer.include) == 0:
                include_layer = True
            else:
                for layer_phase in layer.include:
                    if caffe.TEST == layer_phase.phase:
                        include_layer = True

            # exclude layers not implemented (or implemented in a different fashion)
            if layer.type.lower() not in common_layers:
                include_layer = False

            if include_layer:
                layer_list.append(make_composite_layer(layer))

        # associate activations with computation layers
        for layer in net.layer:
            layer_type = layer.type.lower()
            if layer_type in ['relu', 'prelu', 'elu', 'sigmoid', 'tanh']:
                layer_index = None
                for ii in range(len(layer_list)):
                    if layer.top[0] == layer_list[ii].layer_parm.top[0]:
                        layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        'Activation layer ' + layer.name +
                        ' is not associated with any computation layer.')

        # associate dropout with computation layers
        for layer in net.layer:
            layer_type = layer.type.lower()
            if layer_type == 'dropout':
                layer_index = None
                for ii in range(len(layer_list)):
                    if layer.top[0] == layer_list[ii].layer_parm.top[0]:
                        layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        'Dropout layer ' + layer.name +
                        ' is not associated with any computation layer.')

        # associate softmax with a fully-connected layer
        for layer in net.layer:
            layer_type = layer.type.lower()
            if layer_type in ['softmax', 'softmaxwithloss']:
                layer_index = None
                for ii in range(len(layer_list)):
                    for jj in range(len(layer.bottom)):
                        if layer.bottom[jj] == layer_list[ii].layer_parm.top[0]:
                            layer_index = ii

                if layer_index is not None:
                    layer_list[layer_index].related_layers.append(layer)
                else:
                    raise CaffeParseError(
                        'Softmax layer ' + layer.name +
                        ' is not associated with any fully-connected layer.')

        # determine source layer(s) for computation layers
        for ii in range(len(layer_list)):
            for kk in range(len(layer_list[ii].layer_parm.bottom)):
                name = None
                for jj in range(ii):
                    if (layer_list[ii].layer_parm.bottom[kk] ==
                            layer_list[jj].layer_parm.top[0]):
                        name = layer_list[jj].layer_parm.name

                if name:
                    layer_list[ii].source_layer.append(name)

        # associate scale layer with batchnorm layer
        for layer in net.layer:
            if layer.type.lower() == 'scale':
                bn_found = False
                for ii in range(len(layer_list)):
                    if ((layer_list[ii].layer_parm.type.lower() == 'batchnorm') and
                            (layer_list[ii].layer_parm.top[0] == layer.top[0])):
                        layer_list[ii].related_layers.append(layer)
                        bn_found = True
                        break

                if not bn_found:
                    raise CaffeParseError(
                        'Scale layer ' + layer.name +
                        ' is not associated with a batch normalization layer')

        # loop over included layers
        for clayer in layer_list:
            layer_type = clayer.layer_parm.type.lower()
            if layer_type == 'pooling':  # average/max pooling
                sas_code = caffe_pooling_layer(clayer, model_name)
            elif layer_type == 'convolution':  # 2D convolution
                sas_code = caffe_convolution_layer(clayer, model_name)
            elif layer_type == 'batchnorm':  # batch normalization
                sas_code = caffe_batch_normalization_layer(clayer, model_name)
            elif layer_type in ['data', 'memorydata']:  # input layer
                sas_code = caffe_input_layer(clayer, model_name)
            elif layer_type == 'eltwise':  # residual
                sas_code = caffe_residual_layer(clayer, model_name)
            elif layer_type == 'innerproduct':  # fully connected
                sas_code = caffe_full_connect_layer(clayer, model_name)
            else:
                raise CaffeParseError(layer_type +
                                      ' is an unsupported layer type')

            # write SAS code associated with Caffe layer
            if sas_code:
                output_code = output_code + sas_code + '\n\n'

            else:
                raise CaffeParseError(
                    'Unable to generate SAS definition for layer ' +
                    clayer.layer_parm.name)

        # convert from BINARYPROTO to HDF5
        if network_param is not None:
            sas_hdf5 = os.path.join(os.getcwd(), '{}_weights.caffemodel.h5'.format(model_name))
            write_caffe_hdf5(model, layer_list, sas_hdf5)
            print('NOTE: the model weights has been stored in the following file:\n'
              '{}'.format(sas_hdf5))

        return output_code

    except CaffeParseError as err_msg:
        print(err_msg)


# parse parameters for pooling layer and generate equivalent SAS code
def caffe_pooling_layer(clayer, model_name):
    '''
    Extract pooling layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning pooling layer definition

    '''

    layer_parm = clayer.layer_parm

    # list defining PoolingParameter data structure --> keep in sync with caffe.proto
    dstruct = [{'field': 'pool', 'repeated': False},
               {'field': 'pad', 'repeated': False},
               {'field': 'pad_h', 'repeated': False},
               {'field': 'pad_w', 'repeated': False},
               {'field': 'kernel_size', 'repeated': False},
               {'field': 'kernel_h', 'repeated': False},
               {'field': 'kernel_w', 'repeated': False},
               {'field': 'stride', 'repeated': False},
               {'field': 'stride_h', 'repeated': False},
               {'field': 'stride_w', 'repeated': False},
               {'field': 'engine', 'repeated': False},
               {'field': 'global_pooling', 'repeated': False}]

    # read pooling parameters
    pooling_param = getattr(layer_parm, 'pooling_param', None)
    parms = {}
    if (pooling_param is not None):
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = ('extract_repeated_attr' + '(pooling_param,\'' + 
                            dstruct[ii]['field'] + '\')')
            else:
                code_str = ('extract_attr' + '(pooling_param,\'' + 
                            dstruct[ii]['field'] + '\')')

            parms[dstruct[ii]['field']] = eval(code_str)
    else:
        raise CaffeParseError('No pooling parameters given')

    # define parameters needed by SAS pooling layer

    # pooling type
    if parms['pool'] == 0:
        pool_type = 'max'
    elif parms['pool'] == 1:
        pool_type = 'mean'
    else:
        raise CaffeParseError('Invalid pooling type specified for layer = ' +
                              layer_parm.name)

    # stride (vertical)
    if parms['stride_h'] > 0:
        tmp_stride_h = parms['stride_h']
    else:
        if parms['stride'] == 0:
            tmp_stride_h = 1
        else:
            tmp_stride_h = parms['stride']

    # stride (horizontal)
    if parms['stride_w'] > 0:
        tmp_stride_w = parms['stride_w']
    else:
        if parms['stride'] == 0:
            tmp_stride_w = 1
        else:
            tmp_stride_w = parms['stride']

    # horizontal/vertical stride must agree
    if tmp_stride_w != tmp_stride_h:
        raise CaffeParseError('Horizontal/vertical strides do not agree '
                              'for layer = ' + layer_parm.name)
    else:
        common_stride = tmp_stride_w

    # height of kernel
    if parms['kernel_h'] > 0:
        height = parms['kernel_h']
    else:
        if parms['kernel_size'] == 0:
            raise CaffeParseError('Unable to set kernel height for layer = ' +
                                  layer_parm.name)
        else:
            height = parms['kernel_size']

    # width of kernel
    if parms['kernel_w'] > 0:
        width = kernel_w
    else:
        if parms['kernel_size'] == 0:
            raise CaffeParseError('Unable to set kernel width for layer = ' +
                                  layer_parm.name)
        else:
            width = parms['kernel_size']

    # determine dropout
    dropout = extract_dropout(clayer)
    if dropout is None:
        dropout = 0

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if num_layers != 1:
        raise CaffeParseError('Pooling layer requires one input layer, ' +
                              str(num_layers) + ' provided')

    return write_pooling_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                               width=str(width), height=str(height),
                               stride=str(common_stride), type=pool_type,
                               dropout=str(dropout), src_layer=source_layer)


# parse parameters for convolution layer and generate equivalent SAS code
def caffe_convolution_layer(clayer, model_name):
    '''
    Extract convolution layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning convolution layer definition

    '''

    layer_parm = clayer.layer_parm

    # list defining ConvolutionParameter data structure --> keep in sync with caffe.proto
    dstruct = [{'field': 'num_output', 'repeated': False},
               {'field': 'bias_term', 'repeated': False},
               {'field': 'pad', 'repeated': True},
               {'field': 'kernel_size', 'repeated': True},
               {'field': 'stride', 'repeated': True},
               {'field': 'dilation', 'repeated': True},
               {'field': 'pad_h', 'repeated': False},
               {'field': 'pad_w', 'repeated': False},
               {'field': 'kernel_h', 'repeated': False},
               {'field': 'kernel_w', 'repeated': False},
               {'field': 'stride_h', 'repeated': False},
               {'field': 'stride_w', 'repeated': False},
               {'field': 'group', 'repeated': False},
               {'field': 'weight_filler', 'repeated': False},
               {'field': 'bias_filler', 'repeated': False},
               {'field': 'engine', 'repeated': False},
               {'field': 'axis', 'repeated': False},
               {'field': 'force_nd_im2col', 'repeated': False}]

    # read convolution parameters
    convolution_param = getattr(layer_parm, 'convolution_param', None)
    parms = {}
    if convolution_param is not None:
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = ('extract_repeated_attr' + '(convolution_param,\'' + 
                            dstruct[ii]['field'] + '\')')
            else:
                code_str = ('extract_attr' + '(convolution_param,\'' + 
                            dstruct[ii]['field'] + '\')')

            parms[dstruct[ii]['field']] = eval(code_str)
    else:
        raise CaffeParseError('No convolution parameters given')
    
    # define parameters needed by SAS convolution layer
    # bias
    if parms['bias_term']:
        nobias = 'False'
    else:
        nobias = 'True'

    # number of output layers
    if parms['num_output'] == 0:
        raise CaffeParseError('num_output not provided for layer = ' +
                              layer_parm.name)
    else:
        num_output = parms['num_output']

    # stride (vertical)
    if parms['stride_h'] > 0:
        tmp_stride_h = parms['stride_h']
    else:
        if parms['stride'] == 0:
            tmp_stride_h = 1
        else:
            tmp_stride_h = parms['stride']

    # stride (horizontal)
    if parms['stride_w'] > 0:
        tmp_stride_w = parms['stride_w']
    else:
        if (parms['stride'] == 0):
            tmp_stride_w = 1
        else:
            tmp_stride_w = parms['stride']

    # horizontal/vertical stride must agree
    if tmp_stride_w != tmp_stride_h:
        raise CaffeParseError('Horizontal/vertical strides do not '
                              'agree for layer = ' + layer_parm.name)
    else:
        common_stride = tmp_stride_w

    # height of kernel
    if parms['kernel_h'] > 0:
        height = parms['kernel_h']
    else:
        if parms['kernel_size'] == 0:
            raise CaffeParseError('Unable to set kernel height for layer = ' +
                                  layer_parm.name)
        else:
            height = parms['kernel_size']

    # width of kernel
    if parms['kernel_w'] > 0:
        width = parms['kernel_w']
    else:
        if parms['kernel_size'] == 0:
            raise CaffeParseError('Unable to set kernel width for layer = ' +
                                  layer_parm.name)
        else:
            width = parms['kernel_size']

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if num_layers != 1:
        raise CaffeParseError('Convolution layer requires one input layer, ' +
                              str(num_layers) + ' provided')

    # determine activation
    act = extract_activation(clayer, 'convolution')

    # determine dropout
    dropout = extract_dropout(clayer)
    if dropout is None:
        dropout = 0

    return write_convolution_layer(model_name=model_name,
                                   layer_name=clayer.layer_parm.name,
                                   nfilters=str(num_output), width=str(width),
                                   height=str(height), stride=str(common_stride),
                                   nobias=nobias, activation=act, dropout=str(dropout),
                                   src_layer=source_layer)


# parse parameters for batch normalization layer and generate equivalent SAS code
def caffe_batch_normalization_layer(clayer, model_name):
    '''
    Extract batch normalization layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning batch normalization layer definition

    '''

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if (num_layers != 1):
        raise CaffeParseError(
            'Batch normalization layer requires one input layer, ' +
            str(num_layers) + ' provided')

    # determine activation
    act = extract_activation(clayer, 'batchnorm')

    return write_batch_norm_layer(model_name=model_name,
                                  layer_name=clayer.layer_parm.name,
                                  activation=act, src_layer=source_layer)


# parse parameters for input layer and generate equivalent SAS code
def caffe_input_layer(clayer, model_name):
    '''
    Extract input layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning input layer definition

    '''

    layer_parm = clayer.layer_parm

    # read scaling parameter
    transform_param = getattr(layer_parm, 'transform_param', None)
    if transform_param is not None:
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
        print('WARNING: unable to provide parameters for image data format')

    return write_input_layer(model_name=model_name, layer_name=layer_parm.name,
                             channels=str(channels), width=str(width),
                             height=str(height), scale=str(scale))


# parse parameters for residual layer and generate equivalent SAS code
def caffe_residual_layer(clayer, model_name):
    '''
    Extract residual layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning residual layer definition

    '''

    layer_parm = clayer.layer_parm

    # list defining EltwiseParameter data structure --> keep in sync with caffe.proto
    dstruct = [{'field': 'operation', 'repeated': False},
               {'field': 'coeff', 'repeated': True},
               {'field': 'stable_product_grad', 'repeated': False}]

    # read eltwise parameters
    eltwise_param = getattr(layer_parm, 'eltwise_param', None)
    parms = {}
    if eltwise_param is not None:
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = ('extract_repeated_attr' + '(eltwise_param,\'' + 
                            dstruct[ii]['field'] + '\')')
            else:
                code_str = ('extract_attr' + '(eltwise_param,\'' + 
                            dstruct[ii]['field'] + '\')')

            parms[dstruct[ii]['field']] = eval(code_str)
    else:
        raise CaffeParseError('No eltwise parameters given')

    # determine whether operation specified is valid
    if parms['operation'] != 1:
        raise CaffeParseError('Element-wise operation not supported')

    # determine activation
    act = extract_activation(clayer, 'residual')

    # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if num_layers < 2:
        raise CaffeParseError(
            'Residual layer requires two or more input layers, ' +
            str(num_layers) + ' provided')

    return write_residual_layer(model_name=model_name, layer_name=clayer.layer_parm.name,
                                activation=act, src_layer=source_layer)


# parse parameters for fully connected layer and generate equivalent SAS code
def caffe_full_connect_layer(clayer, model_name):
    '''
    Extract fully-connected layer parameters from LayerParameter object

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    model_name : string
       Deep learning model name.

    Returns
    -------
    String value with SAS deep learning fully-connected layer definition

    '''

    layer_parm = clayer.layer_parm

    # list defining InnerProductParameter data structure --> keep in sync with caffe.proto
    dstruct = [{'field': 'num_output', 'repeated': False},
               {'field': 'bias_term', 'repeated': False},
               {'field': 'weight_filler', 'repeated': False},
               {'field': 'bias_filler', 'repeated': False},
               {'field': 'axis', 'repeated': False},
               {'field': 'transpose', 'repeated': False}]

    # read inner product parameters
    inner_product_param = getattr(layer_parm, 'inner_product_param', None)
    parms = {}
    if inner_product_param is not None:
        for ii in range(len(dstruct)):
            if (dstruct[ii]['repeated']):
                code_str = ('extract_repeated_attr' + '(inner_product_param,\'' + 
                            dstruct[ii]['field'] + '\')')
            else:
                code_str = ('extract_attr' + '(inner_product_param,\'' + 
                            dstruct[ii]['field'] + '\')')

            parms[dstruct[ii]['field']] = eval(code_str)
    else:
        raise CaffeParseError('No inner_product parameters given')

    # define parameters needed by SAS fully-connected layer

    # bias
    if parms['bias_term']:
        nobias = 'False'
    else:
        nobias = 'True'

    # number of output neurons
    if parms['num_output'] > 0:
        num_neurons = parms['num_output']
    else:
        raise CaffeParseError('Number of output neurons not specified '
                              'for layer = , ' + layer_parm.name)

    # check axis setting
    if parms['axis'] != 1:
        raise CaffeParseError('axis = , ' + str(parms['axis']) + ' is not supported')

    # check transpose setting
    if parms['transpose']:
        raise CaffeParseError('transpose = , ' + str(parms['transpose']) +
                              ' is not supported')

    # determine activation
    act = extract_activation(clayer, 'innerproduct')

    # determine layer type
    if act == 'softmax':
        fc_type = 'output'
    else:
        fc_type = 'fullconnect'

    # determine dropout
    dropout = extract_dropout(clayer)
    if (dropout is None):
        dropout = 0

        # determine source layer(s)
    source_layer, num_layers = extract_source_layers(clayer)
    if num_layers != 1:
        raise CaffeParseError('Fully connected layer requires one input layer, ' +
                              str(num_layers) + ' provided')

    return write_full_connect_layer(model_name=model_name,
                                    layer_name=layer_parm.name,
                                    nrof_neurons=str(num_neurons),
                                    nobias=nobias, activation=act,
                                    type=fc_type, dropout=str(dropout),
                                    src_layer=source_layer)


class CompositeLayer(object):
    '''
    Composite layer

    A composite layer is one that consists of common SAS/Caffe
    computation layers along with Caffe layers that share the same top
    blob as the computation layer.

    Parameters
    ----------
    layer_parm :
        LayerParameter object (mirrors Google protobuf definition).

    '''

    def __init__(self, layer_parm):
        self.source_layer = []
        self.layer_parm = layer_parm
        self.related_layers = []


def make_composite_layer(layer_parm):
    '''
    Generate a CompositeLayer object

    Parameters
    ----------
    layer_parm :
        LayerParameter object (mirrors Google protobuf definition).

    Returns
    -------
    :class:`CompositeLayer`

    '''
    return CompositeLayer(layer_parm)


# map Caffe activation layer types to SAS activation types
def map_caffe_activation(layer_name, layer_type, act_type):
    '''
    Map Caffe activation function(s) to SAS activation function(s)

    Parameters
    ----------
    layer_name : string
       Layer name.
    layer_type : string
       Caffe layer type.
    act_type : string
       Caffe activation type.

    Returns
    -------
    SAS activation type

    '''

    # convolution layer
    if layer_type in ['convolution', 'batchnorm', 'residual']:
        map_dict = {'elu': 'elu', 'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid'}
    elif layer_type == 'innerproduct':
        map_dict = {'softmax': 'softmax', 'elu': 'elu', 'relu': 'relu',
                    'tanh': 'tanh', 'sigmoid': 'sigmoid', 'softmaxwithloss': 'softmax'}
    else:
        raise CaffeParseError('SAS does not support activation functions for layer ' +
                              layer_name)

    if act_type in map_dict.keys():
        act_func = map_dict[act_type]
        if act_func is None:
            raise CaffeParseError('Activation function ' + act_type + ' not supported')
    else:
        raise CaffeParseError('Unknown Caffe activation function = ' + act_type)

    return act_func


# extract activation from layer definition
def extract_activation(clayer, layer_type):
    '''
    Extract Caffe activation function from Caffe layer(s) sharing a common top blob

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.
    layer_type : string
       Caffe layer type.

    Returns
    -------
    SAS activation function [default = identity]

    '''
    act = None
    if len(clayer.related_layers) > 0:
        for ii in range(len(clayer.related_layers)):
            act_type = clayer.related_layers[ii].type.lower()
            if act_type in caffe_activation_types:
                if (act is None):
                    act = map_caffe_activation(clayer.layer_parm.name,
                                               layer_type, act_type)
                else:
                    raise CaffeParseError('More than one activation associated '
                                          'with layer = ' + clayer.layer_parm.name)

    if act is None:
        act = 'identity'

    return act


# extract dropout parameter
def extract_dropout(clayer):
    '''
    Extract dropout parameter from Caffe dropout layer

    Parameters
    ----------
    clayer : CompositeLayer object
       Layer parameters.

    Returns
    -------
    Caffe dropout parameter [default = 0.0]

    '''
    dropout = None
    dropout_ratio = 0.0
    if len(clayer.related_layers) > 0:
        for ii in range(len(clayer.related_layers)):
            layer_type = clayer.related_layers[ii].type.lower()
            if layer_type == 'dropout':
                if dropout is None:
                    # read dropout parameters --> only one variable in
                    # DropoutParameter message, so no intervening layer_param wrapper
                    dropout_param = getattr(clayer.related_layers[ii],
                                            'dropout_param', None)
                    if dropout_param is not None:
                        # dropout ratio
                        dropout_ratio = getattr(dropout_param, 'dropout_ratio', 0.0)
                    else:
                        raise CaffeParseError('No dropout parameters given')
                else:
                    raise CaffeParseError(
                        'More than one dropout layer associated with layer = ' +
                        clayer.related_layers[ii].layer_parm.name)
    return dropout_ratio


# determine source layer(s) for a given computation layer
def extract_source_layers(clayer):
    '''
    Construct a string representation of the layer name(s) of all the source layers

    Parameters
    ----------
    clayer : CompositeLayer
       Layer parameters.

    Returns
    -------
    string
        String representation of Python list

    '''
    source_layer = []
    num_layers = len(clayer.source_layer)
    for ii in range(num_layers):
        source_layer.append(clayer.source_layer[ii])
    return repr(source_layer), num_layers


# extract value from repeated container object (only first returned)
def extract_repeated_attr(param, field):
    '''
    Extract a particular field defined as a RepeatedContainer

    Parameters
    ----------
    param : parameter object
       Various parameter objects defined by Google messages.
    field : string
       Parameter field.

    Notes
    -----
    Only the first value is returned.

    Returns
    -------
    string or None
        Field value or None if parameter or field doesn't exist.

    '''
    tmpval = getattr(param, field, None)
    if tmpval is not None:
        if isinstance(tmpval, (float, int, bool)):
            y = tmpval
        elif len(tmpval) > 0:
            y = tmpval[0]
        else:
            y = None
        return y
    else:
        return None


# extract value
def extract_attr(param, field):
    '''
    Extract a particular field from a parameter object

    Parameters
    ----------
    param : parameter object
       Various parameter objects defined by Google messages.
    field : string
       Parameter field.

    Returns
    -------
    string or None
        Field value or None if parameter or field doesn't exist.

    '''
    tmpval = getattr(param, field, None)
    if tmpval is not None:
        return tmpval
    else:
        return None
