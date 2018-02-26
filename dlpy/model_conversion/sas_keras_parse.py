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

'''Convert keras model to sas models.'''

import os
import sys

from keras import backend as K

from .write_keras_model_parm import write_keras_hdf5
from .write_sas_code import (write_input_layer, write_convolution_layer,
                             write_batch_norm_layer, write_pooling_layer,
                             write_residual_layer, write_full_connect_layer,
                             write_main_entry)

computation_layer_classes = ['averagepooling2d', 'maxpooling2d', 'conv2d',
                             'dense', 'batchnormalization', 'add']
dropout_layer_classes = ['averagepooling2d', 'maxpooling2d', 'conv2d', 'dense']


class KerasParseError(ValueError):
    '''
    Used to indicate an error in parsing Keras model definition

    '''


# def keras_to_sas(module_name, model_name, model_args, sas_file_name, input_shape=None):
def keras_module_to_sas(module_name, model_name, model_args, sas_file_name):
    '''
    Function to generate a SAS deep learning model from Keras definition

    Parameters:

    ----------
    module_name : [string]
       Name of module containing function definition of deep learning model
    model_name : [string]
       Name of function defining deep learning model
    model_args : [string]
       Arguments to pass to Keras model instantiation
    sas_file_name : [string]
       Fully qualified file name of SAS deep learning Python model definition
    input_shape : [tuple, optional --> not supported yet]
       User-defined input shape in channels first format (e.g. (C, H, W))

    Returns
    -------
    Keras model object
    '''

    # open output file
    try:
        fout = open(sas_file_name, 'w')
    except IOError:
        sys.exit('Unable to create file ' + sas_file_name)

    # instantiate the model
    exec('from ' + module_name + ' import *')
    model = eval(model_name + '(' + model_args + ')')

    try:

        if model:

            # # check that input shape (if given) has correct dimensions
            # if (input_shape is None):
            # pass
            # else:  # revisit model with custome input shape
            # if (len(input_shape) != 3):
            # raise KerasParseError('ERROR: input shape specified incorrectly')
            # else:
            # decim_factor = 1.0
            # for layer in model.layers:
            # class_name = layer.__class__.__name__.lower()
            # if (class_name == 'inputlayer'):
            # print(layer.batch_input_shape)
            # elif (class_name == 'averagepooling2d'):
            # print(layer.get_config())
            # print(layer.get_input_shape_at(0))

            # sys.exit('Done')

            # extract all activation functions, source layers, and dropout for layers
            # that are not separately implemented in SAS deep learning actions
            layer_activation = {}
            src_layer = {}
            layer_dropout = {}
            for layer in model.layers:
                class_name = layer.__class__.__name__.lower()
                if (class_name in computation_layer_classes):
                    comp_layer_name = find_previous_computation_layer(
                        model, layer.name, computation_layer_classes)
                    source_str = make_source_str(comp_layer_name)
                    src_layer.update({layer.name: source_str})
                elif (class_name == 'activation'):
                    tmp_name = find_previous_computation_layer(
                        model, layer.name, computation_layer_classes)
                    tmp_act = extract_activation(layer)
                    layer_activation.update({tmp_name[0]: tmp_act})
                elif (class_name == 'dropout'):
                    tmp = find_next_computation_layer(model, layer, dropout_layer_classes)
                    dconfig = layer.get_config()
                    layer_dropout.update({tmp: dconfig['rate']})

            # if first layer is not an input layer, generate the correct
            # input layer code for a SAS deep learning model
            layer = model.layers[0]
            if (layer.__class__.__name__.lower() != 'inputlayer'):
                sas_code = keras_input_layer(layer, model_name, False)
                # write SAS code for input layer
                if sas_code:
                    fout.write(sas_code + '\n\n')
                else:
                    raise KerasParseError('ERROR: unable to generate an input layer')

            # extract layers and apply activation functions as needed
            for layer in model.layers:
                class_name = layer.__class__.__name__.lower()

                sas_code = None

                # determine activation function
                if (class_name in ['conv2d', 'batchnormalization', 'add', 'dense']):
                    if (layer.name in layer_activation.keys()):
                        act_func = layer_activation[layer.name]
                    else:
                        act_func = None
                else:
                    act_func = None

                # average/max pooling
                if (class_name in ['averagepooling2d', 'maxpooling2d']):
                    sas_code = keras_pooling_layer(layer, model_name, class_name,
                                                   src_layer, layer_dropout)
                # 2D convolution
                elif (class_name == 'conv2d'):
                    sas_code = keras_convolution_layer(layer, model_name, act_func,
                                                       src_layer, layer_dropout)
                # batch normalization
                elif (class_name == 'batchnormalization'):
                    sas_code = keras_batchnormalization_layer(layer, model_name,
                                                              act_func, src_layer)
                # input layer
                elif (class_name == 'inputlayer'):
                    sas_code = keras_input_layer(layer, model_name, True)
                # add
                elif (class_name == 'add'):
                    sas_code = keras_residual_layer(layer, model_name,
                                                    act_func, src_layer)
                elif (class_name in ['activation', 'flatten', 'dropout']):
                    pass
                # fully connected
                elif (class_name == 'dense'):
                    sas_code = keras_full_connect_layer(layer, model_name, act_func,
                                                        src_layer, layer_dropout)
                else:
                    print('WARNING: ' + class_name + ' is an unsupported layer '
                                                     'type - your SAS model may be incomplete')

                # write SAS code associated with Keras layer
                if sas_code:
                    fout.write(sas_code + '\n\n')
                elif (class_name not in ['activation', 'flatten', 'dropout']):
                    print('WARNING: unable to generate SAS definition '
                          'for layer ' + class_name)

        else:
            raise KerasParseError('ERROR: Unable to instantiate Keras model')

    except KerasParseError as err_msg:
        print(err_msg)
    except Exception as err_msg:
        print(err_msg)
    finally:
        sas_code = write_main_entry(model_name)
        fout.write(sas_code + '\n')
        fout.close()
        if model:
            return model


def keras_to_sas(model, model_name=None):
    output_code = ''
    layer_activation = {}
    src_layer = {}
    layer_dropout = {}
    if model_name is None:
        model_name = model.name
    for layer in model.layers:
        class_name = layer.__class__.__name__.lower()
        if (class_name in computation_layer_classes):
            comp_layer_name = find_previous_computation_layer(
                model, layer.name, computation_layer_classes)
            source_str = make_source_str(comp_layer_name)
            src_layer.update({layer.name: source_str})
        elif (class_name == 'activation'):
            tmp_name = find_previous_computation_layer(
                model, layer.name, computation_layer_classes)
            tmp_act = extract_activation(layer)
            layer_activation.update({tmp_name[0]: tmp_act})
        elif (class_name == 'dropout'):
            tmp = find_next_computation_layer(model, layer, dropout_layer_classes)
            dconfig = layer.get_config()
            layer_dropout.update({tmp: dconfig['rate']})

    # if first layer is not an input layer, generate the correct
    # input layer code for a SAS deep learning model
    layer = model.layers[0]
    if (layer.__class__.__name__.lower() != 'inputlayer'):
        sas_code = keras_input_layer(layer, model_name, False)
        # write SAS code for input layer
        if sas_code:
            output_code = output_code + sas_code + '\n\n'
        else:
            raise KerasParseError('ERROR: unable to generate an input layer')

    # extract layers and apply activation functions as needed
    for layer in model.layers:
        class_name = layer.__class__.__name__.lower()

        sas_code = None

        # determine activation function
        if (class_name in ['conv2d', 'batchnormalization', 'add', 'dense']):
            if (layer.name in layer_activation.keys()):
                act_func = layer_activation[layer.name]
            else:
                act_func = None
        else:
            act_func = None

        # average/max pooling
        if (class_name in ['averagepooling2d', 'maxpooling2d']):
            sas_code = keras_pooling_layer(layer, model_name, class_name,
                                           src_layer, layer_dropout)
        # 2D convolution
        elif (class_name == 'conv2d'):
            sas_code = keras_convolution_layer(layer, model_name, act_func,
                                               src_layer, layer_dropout)
        # batch normalization
        elif (class_name == 'batchnormalization'):
            sas_code = keras_batchnormalization_layer(layer, model_name,
                                                      act_func, src_layer)
        # input layer
        elif (class_name == 'inputlayer'):
            sas_code = keras_input_layer(layer, model_name, True)
        # add
        elif (class_name == 'add'):
            sas_code = keras_residual_layer(layer, model_name,
                                            act_func, src_layer)
        elif (class_name in ['activation', 'flatten', 'dropout']):
            pass
        # fully connected
        elif (class_name == 'dense'):
            sas_code = keras_full_connect_layer(layer, model_name, act_func,
                                                src_layer, layer_dropout)
        else:
            print('WARNING: ' + class_name + ' is an unsupported layer '
                                             'type - your SAS model may be incomplete')

        # write SAS code associated with Keras layer
        if sas_code:
            output_code = output_code + sas_code + '\n\n'
        elif (class_name not in ['activation', 'flatten', 'dropout']):
            print('WARNING: unable to generate SAS definition '
                  'for layer ' + class_name)

    return output_code


# create SAS pooling layer
def keras_pooling_layer(layer, model_name, class_name, src_layer, layer_dropout):
    '''
    Function to extract pooling layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Pooling layer
    model_name : [string]
       Deep learning model name
    class_name : [string]
       Layer class
    src_layer : [list of Layer objects]
       Layer objects corresponding to source layer(s) for
       pooling layer
    layer_dropout : [dictionary]
       Dictionary containing dropout layer names (keys) and dropout rates (values)

    Returns
    -------
    String value with SAS deep learning pooling layer definition
    '''
    config = layer.get_config()
    strides = config['strides']
    pool_size = config['pool_size']

    # pooling size
    width, height = pool_size

    # stride
    if (strides[0] == strides[1]):
        step = strides[0]
    else:
        raise KerasParseError('ERROR: unequal strides in vertical/horizontal '
                              'directions for pooling layer')

    # pooling type
    if (class_name == 'averagepooling2d'):
        type = 'mean'
    elif (class_name == 'maxpooling2d'):
        type = 'max'
    else:
        raise KerasParseError('ERROR: Pooling type ' + class_name +
                              ' is not supported yet')

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = src_layer[layer.name]
    else:
        raise KerasParseError('ERROR: unable to determine source layer for '
                              'pooling layer = ' + layer.name)

    # set dropout
    if (layer.name in layer_dropout.keys()):
        dropout = layer_dropout[layer.name]
    else:
        dropout = 0.0

    return write_pooling_layer(model_name=model_name, layer_name=layer.name,
                               width=str(width), height=str(height), stride=str(step),
                               type=type, dropout=str(dropout), src_layer=source_str)


# create SAS 2D convolution layer
def keras_convolution_layer(layer, model_name, act_func, src_layer, layer_dropout):
    '''
    Function to extract convolution layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Convolution layer
    model_name : [string]
       Deep learning model name
    act_func : [string]
       Keras activation function
    src_layer : [list of Layer objects]
       Layer objects corresponding to source layer(s) for
       convolution layer
    layer_dropout : [dictionary]
       Dictionary containing dropout layer names (keys) and dropout rates (values)

    Returns
    -------
    String value with SAS deep learning convolution layer definition
    '''
    config = layer.get_config()

    strides = config['strides']
    kernel_size = config['kernel_size']

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    # kernel size
    width, height = kernel_size

    # stride
    if (strides[0] == strides[1]):
        step = strides[0]
    else:
        raise KerasParseError('ERROR: unequal strides in vertical/horizontal '
                              'directions for convolution layer')

    # bias term
    if (config['use_bias']):
        bias_str = 'False'
    else:
        bias_str = 'True'

    # number of filters
    nrof_filters = config['filters']

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = src_layer[layer.name]
    else:
        raise KerasParseError('ERROR: unable to determine source layer for '
                              'convolution layer = ' + layer.name)

    # set dropout
    if (layer.name in layer_dropout.keys()):
        dropout = layer_dropout[layer.name]
    else:
        dropout = 0.0

    return write_convolution_layer(model_name=model_name, layer_name=layer.name,
                                   nfilters=str(nrof_filters), width=str(width),
                                   height=str(height), stride=str(step),
                                   nobias=bias_str, activation=layer_act_func,
                                   dropout=str(dropout), src_layer=source_str)


# create SAS batch normalization layer
def keras_batchnormalization_layer(layer, model_name, act_func, src_layer):
    '''
    Function to extract batch normalization layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Batch nornalization layer
    model_name : [string]
       Deep learning model name
    act_func : [string]
       Keras activation function
    src_layer : [list of Layer objects]
       Layer objects corresponding to source layer(s) for
       batch normalization layer

    Returns
    -------
    String value with SAS deep learning batch normalization layer definition
    '''
    config = layer.get_config()

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = src_layer[layer.name]
    else:
        raise KerasParseError('ERROR: unable to determine source layer for '
                              'batch normalization layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    return write_batch_norm_layer(model_name=model_name, layer_name=layer.name,
                                  activation=layer_act_func, src_layer=source_str)


# create SAS input layer
def keras_input_layer(layer, model_name, input_layer):
    '''
    Function to extract input layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Input layer
    model_name : [string]
       Deep learning model name
    input_layer : [Boolean]
       Indicate whether layer name given (True) or not (False)

    Returns
    -------
    String value with SAS deep learning input layer definition
    '''
    config = layer.get_config()

    if (K.image_data_format() == 'channels_first'):
        dummy, C, H, W = config['batch_input_shape']
    else:
        dummy, H, W, C = config['batch_input_shape']

    # generate name based on whether layer is actually an input layer
    # TODO: input_name is never used
    if (input_layer):
        input_name = config['name']
    else:
        input_name = 'data'

    # TBD: fix scale, now default value for scale
    scale = 1.0

    return write_input_layer(model_name=model_name, layer_name=layer.name,
                             channels=str(C), width=str(W),
                             height=str(H), scale=str(scale))


# create SAS residual layer
def keras_residual_layer(layer, model_name, act_func, src_layer):
    '''
    Function to extract residual layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Add layer
    model_name : [string]
       Deep learning model name
    act_func : [string]
       Keras activation function
    src_layer : [list of Layer objects]
       Layer objects corresponding to source layer(s) for
       residual layer

    Returns
    -------
    String value with SAS deep learning residual layer definition
    '''
    config = layer.get_config()

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = src_layer[layer.name]
    else:
        raise KerasParseError('ERROR: unable to determine source layers for '
                              'residual layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    return write_residual_layer(model_name=model_name, layer_name=layer.name,
                                activation=layer_act_func, src_layer=source_str)


# create SAS fully connected layer
def keras_full_connect_layer(layer, model_name, act_func, src_layer, layer_dropout):
    '''
    Function to extract fully connected layer parameters from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Fully connected layer
    model_name : [string]
       Deep learning model name
    act_func : [string]
       Keras activation function
    src_layer : [list of Layer objects]
       Layer objects corresponding to source layer(s) for
       fully connected layer
    layer_dropout : [dictionary]
       Dictionary containing dropout layer names (keys) and dropout rates (values)

    Returns
    -------
    String value with SAS deep learning fully connected layer definition
    '''
    config = layer.get_config()

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    # check whether SOFTMAX or other type of activation
    if (layer_act_func.lower() == 'softmax'):
        layer_type = 'output'
    else:
        layer_type = 'fullconnect'

    # number of neurons
    nrof_neurons = config['units']

    # bias term
    if (config['use_bias']):
        bias_str = 'False'
    else:
        bias_str = 'True'

        # set dropout
    if (layer.name in layer_dropout.keys()):
        dropout = layer_dropout[layer.name]
    else:
        dropout = 0.0

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = src_layer[layer.name]
    else:
        raise KerasParseError('ERROR: unable to determine source layer for '
                              'fully connected layer = ' + layer.name)

    return write_full_connect_layer(model_name=model_name, layer_name=layer.name,
                                    nrof_neurons=str(nrof_neurons), nobias=bias_str,
                                    activation=layer_act_func, type=layer_type,
                                    dropout=str(dropout), src_layer=source_str)


def map_keras_activation(layer, act_func):
    '''
    Function to map Keras activation function(s) to SAS
    activation function(s)

    Parameters:

    ----------
    layer : [Layer object]
       Current layer definition
    act_func : [string]
       Keras activation function

    Returns
    -------
    SAS activation type
    '''
    class_name = layer.__class__.__name__.lower()
    # convolution layer
    if (class_name in ['conv2d', 'batchnormalization']):
        map_dict = {'softmax': None, 'elu': 'elu', 'selu': None,
                    'softplus': 'softplus', 'softsign': None,
                    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                    'hard_sigmoid': None, 'linear': 'identity'}
    elif (class_name == 'dense'):
        map_dict = {'softmax': 'softmax', 'elu': 'elu', 'selu': None,
                    'softplus': 'softplus', 'softsign': None,
                    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                    'hard_sigmoid': None, 'linear': 'identity'}
    elif (class_name == 'add'):
        map_dict = {'softmax': None, 'elu': None, 'selu': None,
                    'softplus': None, 'softsign': None, 'relu': 'relu',
                    'tanh': None, 'sigmoid': None, 'hard_sigmoid': None,
                    'linear': 'identity'}
    else:
        raise KerasParseError('SAS does not support activation functions '
                              'for layer ' + layer.name)

    if (act_func.lower() in map_dict.keys()):
        sas_act_func = map_dict[act_func.lower()]
        if not sas_act_func:
            raise KerasParseError('Activation function ' + act_func + ' not supported')
    else:
        raise KerasParseError('Unknown Keras activation function = ' + act_func)

    return sas_act_func


# extract parameters from activation layer
def extract_activation(layer):
    '''
    Function to extract activation from layer
    definition object

    Parameters:

    ----------
    layer : [Layer object]
       Activation layer

    Returns
    -------
    String value with Keras activation function
    '''
    config = layer.get_config()

    return config['activation']


def find_next_computation_layer(model, layer, computation_layer_list):
    '''
    Function to extract the name of the computation layer
    following the current layer

    Parameters:

    ----------
    model : [Model object]
       Keras deep learning model
    layer : [Layer object]
       Current layer object
    computation_layer_list : [list]
       List of computation layers supported by SAS

    Returns
    -------
    String value with name of next computation layer
    '''

    if (len(layer._outbound_nodes) > 1):
        raise KerasParseError('Unable to determine next computation layer '
                              'for layer = ' + layer.name)
    else:
        node_config = layer._outbound_nodes[0].get_config()
        layer_name = node_config['outbound_layer']
        tmp_layer = model.get_layer(name=layer_name)
        while (tmp_layer.__class__.__name__.lower() not in computation_layer_list):
            if (len(tmp_layer._outbound_nodes) > 1):
                raise KerasParseError('Unable to determine next computation layer '
                                      'for layer = ' + layer.name)
                break
            else:
                node_config = tmp_layer._outbound_nodes[0].get_config()
                layer_name = node_config['outbound_layer']
                tmp_layer = model.get_layer(name=layer_name)

    return layer_name


def find_previous_computation_layer(model, layer_name, computation_layer_list):
    '''
    Function to extract the name of the computation layer
    prior to the current layer

    Parameters:

    ----------
    model : [Model object]
       Keras deep learning model
    layer_name : [string]
       Current layer name
    computation_layer_list : [list]
       List of computation layers supported by SAS

    Returns
    -------
    String value with name of previous computation layer
    '''

    layer = model.get_layer(name=layer_name)
    if (len(layer._inbound_nodes) > 1):
        raise KerasParseError('Unable to determine previous computation '
                              'layer(s) for layer = ' + layer_name)
        return None
    else:
        src_layer_name = []
        node_config = layer._inbound_nodes[0].get_config()
        for lname in node_config['inbound_layers']:
            tmp_layer = model.get_layer(name=lname)
            prev_layer = tmp_layer
            while (tmp_layer.__class__.__name__.lower() not in computation_layer_list):
                # check for root node
                node_config = tmp_layer._inbound_nodes[0].get_config()
                if (len(node_config['inbound_layers']) > 1):
                    raise KerasParseError('Unable to determine previous computation '
                                          'layer(s) for layer = ' + layer_name)
                    return None
                elif (len(node_config['inbound_layers']) == 1):
                    tmp_name = node_config['inbound_layers'][0]
                    prev_layer = tmp_layer
                    tmp_layer = model.get_layer(name=tmp_name)
                else:
                    tmp_layer = prev_layer
                    break

            src_layer_name.append(tmp_layer.name)

        return src_layer_name


def make_source_str(layer_name):
    '''
    Function to create a string value representing a
    Python list of source layers

    Parameters:

    ----------
    layer_name : [string]
       List of Python layer names

    Returns
    -------
    String representation of list of Python layer names

    '''
    source_str = []
    for ii in range(len(layer_name)):
        source_str.append(layer_name[ii])
    return repr(source_str)


#########################################################################################
if __name__ == '__main__':
    # check that environment variables set
    if ('KERAS_HDF5_PATH' not in os.environ.keys()):
        err_msg = ('Environment variable KERAS_HDF5_PATH not set.  '
                   'Please set this variable to \n'
                   'point to the directory where HDF5 files are or will be stored \n')
        sys.exit(err_msg)

    if ('KERAS_APPLICATION_PATH' in os.environ.keys()):
        sys.path.append(os.environ['KERAS_APPLICATION_PATH'])
    else:
        err_msg = ('Environment variable KERAS_APPLICATION_PATH not set.  '
                   'Please set this \nvariable to point to the directory where '
                   'model definition files are or \nwill be stored \n')
        sys.exit(err_msg)

    model_name = 'VGG19'
    if (model_name == 'ResNet50'):
        # ResNet-50
        module_name = 'resnet50'
        model_args = 'weights=None'
        hdf5_in = os.path.join(os.environ['KERAS_HDF5_PATH'],
                               'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
    elif (model_name == 'LeNet'):
        # LeNet
        module_name = 'lenet'
        model_args = ''
        hdf5_in = os.path.join(os.environ['KERAS_HDF5_PATH'], 'lenet.h5')
    elif (model_name == 'VGG16'):
        # VGG-16
        module_name = 'vgg16'
        model_args = 'weights=None'
        hdf5_in = os.path.join(os.environ['KERAS_HDF5_PATH'],
                               'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    elif (model_name == 'VGG19'):
        # VGG-19
        module_name = 'vgg19'
        model_args = 'weights=None'
        hdf5_in = os.path.join(os.environ['KERAS_HDF5_PATH'],
                               'vgg19_weights_tf_dim_ordering_tf_kernels.h5')
    else:
        sys.exit('ERROR: Unknown model specified')

    sas_python = os.path.join(os.environ['KERAS_APPLICATION_PATH'], 'sas_model.py')
    sas_hdf5 = os.path.join(os.environ['KERAS_HDF5_PATH'], 'sas_model.h5')

    # model = keras_to_sas(module_name,model_name,model_args,
    #                      sas_python,input_shape=(3,256,256))
    model = keras_to_sas(module_name, model_name, model_args, sas_python)
    write_keras_hdf5(model, hdf5_in, sas_hdf5)
