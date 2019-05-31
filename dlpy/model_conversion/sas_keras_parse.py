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

''' Convert keras model to sas models '''

import os

from keras import backend as K
from distutils.version import StrictVersion
import keras
from dlpy.utils import DLPyError
if StrictVersion( keras.__version__) < '2.1.3':
    raise DLPyError('This keras version ('+keras.__version__+') is not supported, '
                                                             'please use a version >= 2.1.3')

from .model_conversion_utils import replace_forward_slash, remove_layer_wrapper
from .write_keras_model_parm import write_keras_hdf5
from .write_sas_code import (write_input_layer, write_convolution_layer,
                             write_batch_norm_layer, write_pooling_layer,
                             write_residual_layer, write_full_connect_layer,
                             write_concatenate_layer, write_main_entry,
                             write_recurrent_layer)

computation_layer_classes = ['averagepooling2d', 'maxpooling2d', 'conv2d',
                             'dense', 'batchnormalization', 'add', 'concatenate',
                             'globalaveragepooling2d', 'simplernn', 'lstm', 'gru', 
                             'cudnnlstm', 'cudnngru']
dropout_layer_classes = ['averagepooling2d', 'maxpooling2d', 'conv2d', 'dense']


class KerasParseError(ValueError):
    '''
    Used to indicate an error in parsing Keras model definition

    '''


def keras_to_sas(model, rnn_support, model_name=None, offsets=None, std=None, scale=1.0, max_num_frames=-1, verbose=False):
    output_code = ''
    layer_activation = {}
    src_layer = {}
    layer_dropout = {}
    if model_name is None:
        model_name = model.name
    model_type='CNN'
    n_lambda_layer = 0
    for layer in model.layers:
        class_name, sublayers = remove_layer_wrapper(layer)
        for tlayer in sublayers:
            if (class_name in computation_layer_classes) or (class_name == 'zeropadding2d'):
                comp_layer_name = find_previous_computation_layer(model, layer.name, computation_layer_classes)
                source_str = make_source_str(comp_layer_name)
                src_layer.update({tlayer.name: source_str})
            elif class_name == 'activation':
                tmp_name = find_previous_computation_layer(model, layer.name, computation_layer_classes)
                tmp_act = extract_activation(layer)
                layer_activation.update({tmp_name[0]: tmp_act})
            elif class_name == 'dropout':
                tmp = find_next_computation_layer(model, layer, dropout_layer_classes)
                dconfig = layer.get_config()
                layer_dropout.update({tmp: dconfig['rate']})
            # check for RNN model
            if class_name in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
                if rnn_support:
                    model_type = 'RNN'
                else:
                    raise DLPyError('RNN model detected: your Viya deployment does not support '
                                    'importing an RNN model.')
            # check for Lambda layers
            if layer.__class__.__name__.lower() == 'lambda':
                n_lambda_layer = n_lambda_layer + 1
                
    # if first layer is not an input layer, generate the correct
    # input layer code for a SAS deep learning model
    layer = model.layers[0]
    if layer.__class__.__name__.lower() != 'inputlayer':
        sas_code = keras_input_layer(layer, model_name, False, offsets, std, scale, model_type, max_num_frames)
        # write SAS code for input layer
        if sas_code:
            output_code = output_code + sas_code + '\n\n'
        else:
            raise KerasParseError('Unable to generate an input layer')
  
    # only one Lambda layer supported, and it must be the last model layer
    # assumption: CTC loss must be specified for an RNN model using a 
    #             Lambda layer
    ctc_loss = False
    if n_lambda_layer > 0:
        layer = model.layers[-1]
        if (n_lambda_layer == 1) and (layer.__class__.__name__.lower() == 'lambda') and (model_type == 'RNN'):
            ctc_loss = True
            if verbose:
                print('WARNING - detected a Lambda layer terminating the Keras model.  This is assumed to be '
                      'the CTC loss function definition.  If that is incorrect, please revise your Keras model.')
        else:
            raise KerasParseError('Detected one or more Lambda layers. Only 1 Lambda '
                                  'layer is supported for RNN models, and it must be '
                                  'the last layer.')            
            
    # extract layers and apply activation functions as needed
    zero_pad = None
    for layer in model.layers:
        class_name, sublayers = remove_layer_wrapper(layer)
        for tlayer in sublayers:
            sas_code = None
            
            # determine activation function
            if class_name in ['conv2d', 'batchnormalization', 'add', 'dense']:
                if layer.name in layer_activation.keys():
                    act_func = layer_activation[layer.name]
                else:
                    act_func = None
            else:
                act_func = None
                
            # average/max pooling/globalaveragepooling
            if class_name in ['averagepooling2d', 'maxpooling2d', 
                               'globalaveragepooling2d']:
                sas_code = keras_pooling_layer(tlayer, model_name, class_name,
                                               src_layer, layer_dropout, zero_pad)
                zero_pad = None
            # 2D convolution
            elif class_name == 'conv2d':
                sas_code = keras_convolution_layer(tlayer, model_name, act_func,
                                                   src_layer, layer_dropout, zero_pad)
                zero_pad = None
            # batch normalization
            elif class_name == 'batchnormalization':
                sas_code = keras_batchnormalization_layer(tlayer, model_name,
                                                          act_func, src_layer)
            # input layer
            elif class_name == 'inputlayer':
                sas_code = keras_input_layer(tlayer, model_name, True, offsets, std, scale, model_type, max_num_frames)
            # add
            elif class_name == 'add':
                sas_code = keras_residual_layer(tlayer, model_name,
                                                act_func, src_layer)
            elif class_name in ['activation', 'flatten', 'dropout', 'zeropadding2d', 'lambda']:
                pass
            # fully connected
            elif class_name == 'dense':
                sas_code = keras_full_connect_layer(tlayer, model_name, act_func,
                                                    src_layer, layer_dropout, ctc_loss)
            # concatenate
            elif class_name == 'concatenate':
                sas_code = keras_concatenate_layer(tlayer, model_name,
                                                   act_func, src_layer)
            # recurrent
            elif class_name in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
                sas_code = keras_recurrent_layer(tlayer, model_name,
                                                 act_func, src_layer)
            else:
                raise KerasParseError(class_name + ' is an unsupported layer '
                                      'type - model conversion failed')

            # write SAS code associated with Keras layer
            if sas_code:
                output_code = output_code + sas_code + '\n\n'
            # zero-padding
            elif (class_name == 'zeropadding2d'):
                zero_pad = keras_zeropad2d_layer(tlayer, src_layer)
            elif (class_name not in ['activation', 'flatten', 'dropout', 'lambda']):
                if verbose:
                    print('WARNING: unable to generate SAS definition '
                          'for layer ' + tlayer.name)
    return output_code


# create SAS pooling layer
def keras_pooling_layer(layer, model_name, class_name, src_layer, layer_dropout, zero_pad):
    '''
    Extract pooling layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Pooling layer
    model_name : string
       Deep learning model name
    class_name : string
       Layer class
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       pooling layer
    layer_dropout : dict
       Dictionary containing dropout layer names (keys) and dropout rates (values)
    zero_pad : dict
       Dictionary containing padding values derived from zero-padding layer (may be None)

    Returns
    -------
    string
        String value with SAS deep learning pooling layer definition

    '''
    if class_name == 'globalaveragepooling2d':
        strides = (1, 1)
        padding = 0
        pad_height = None
        pad_width = None
        try:
            pool_size = int(layer.input.shape[1]), int(layer.input.shape[2])
        except:
            raise KerasParseError('Unable to determine dimensions for '
                                  'global average pooling layer')
    else:
        config = layer.get_config()
        strides = config['strides']
        pool_size = config['pool_size']
        if zero_pad is not None:
            pad_height = zero_pad['height']
            pad_width = zero_pad['width']
            padding = None
        else:
            pad_height = None
            pad_width = None
            if config['padding'] == 'valid':
                padding = 0
            else:
                padding = None                

    # pooling size
    height, width = pool_size

    # stride
    if (strides[0] == strides[1]):
        step = strides[0]
    else:
        raise KerasParseError('Unequal strides in vertical/horizontal '
                              'directions for pooling layer')

    # pooling type
    if (class_name == 'averagepooling2d' or 
        class_name == 'globalaveragepooling2d'):
        type = 'mean'
    elif (class_name == 'maxpooling2d'):
        type = 'max'
    else:
        raise KerasParseError('Pooling type ' + class_name +
                              ' is not supported yet')

    # extract source layer(s)
    if zero_pad is not None:
        source_str = replace_forward_slash(zero_pad['source_str'])
    else:    
        if (layer.name in src_layer.keys()):
            source_str = replace_forward_slash(src_layer[layer.name])
        else:
            raise KerasParseError('Unable to determine source layer for '
                                  'pooling layer = ' + layer.name)

    # set dropout
    if (layer.name in layer_dropout.keys()):
        dropout = layer_dropout[layer.name]
    else:
        dropout = 0.0
        
    return write_pooling_layer(model_name=model_name, 
                               layer_name=replace_forward_slash(layer.name),
                               width=str(width), height=str(height), stride=str(step),
                               type=type, dropout=str(dropout), src_layer=source_str,
                               padding=str(padding),pad_height=str(pad_height),
                               pad_width=str(pad_width))


# create SAS 2D convolution layer
def keras_convolution_layer(layer, model_name, act_func, src_layer, layer_dropout, zero_pad):
    '''
    Extract convolution layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Convolution layer
    model_name : string
       Deep learning model name
    act_func : string
       Keras activation function
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       convolution layer
    layer_dropout : dict
       Dictionary containing dropout layer names (keys) and dropout rates (values)
    zero_pad : dict
       Dictionary containing padding values derived from zero-padding layer (may be None)

    Returns
    -------
    string
        String value with SAS deep learning convolution layer definition

    '''
    config = layer.get_config()

    strides = config['strides']
    kernel_size = config['kernel_size']
    if zero_pad is not None:
        pad_height = zero_pad['height']
        pad_width = zero_pad['width']
        padding = None
    else:
        pad_height = None
        pad_width = None
        if config['padding'] == 'valid':
            padding = 0
        else:
            padding = None

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    # kernel size
    height, width = kernel_size

    # stride
    if (strides[0] == strides[1]):
        step = strides[0]
    else:
        raise KerasParseError('Unequal strides in vertical/horizontal '
                              'directions for convolution layer')

    # bias term
    if (config['use_bias']):
        bias_str = 'False'
    else:
        bias_str = 'True'

    # number of filters
    nrof_filters = config['filters']

    # extract source layer(s)
    if zero_pad is not None:
        source_str = replace_forward_slash(zero_pad['source_str'])
    else:
        if (layer.name in src_layer.keys()):
            source_str = replace_forward_slash(src_layer[layer.name])
        else:
            raise KerasParseError('Unable to determine source layer for '
                                  'convolution layer = ' + layer.name)

    # set dropout
    if (layer.name in layer_dropout.keys()):
        dropout = layer_dropout[layer.name]
    else:
        dropout = 0.0
        
    return write_convolution_layer(model_name=model_name, 
                                   layer_name=replace_forward_slash(layer.name),
                                   nfilters=str(nrof_filters), width=str(width),
                                   height=str(height), stride=str(step),
                                   nobias=bias_str, activation=layer_act_func,
                                   dropout=str(dropout), src_layer=source_str,
                                   padding=str(padding),pad_height=str(pad_height),
                                   pad_width=str(pad_width))


# create SAS batch normalization layer
def keras_batchnormalization_layer(layer, model_name, act_func, src_layer):
    '''
    Extract batch normalization layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Batch nornalization layer
    model_name : string
       Deep learning model name
    act_func : string
       Keras activation function
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       batch normalization layer

    Returns
    -------
    string
        String value with SAS deep learning batch normalization layer definition

    '''
    config = layer.get_config()

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = replace_forward_slash(src_layer[layer.name])
    else:
        raise KerasParseError('Unable to determine source layer for '
                              'batch normalization layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    return write_batch_norm_layer(model_name=model_name, 
                                  layer_name=replace_forward_slash(layer.name),
                                  activation=layer_act_func, src_layer=source_str)


# create SAS input layer
def keras_input_layer(layer, model_name, input_layer, offsets, std, scale, model_type, max_num_frames):
    '''
    Extract input layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Input layer
    model_name : string
       Deep learning model name
    input_layer : boolean
       Indicate whether layer name given (True) or not (False)
    offsets : list or None
        Specifies the values to be subtracted from the pixel values
        of the input data, used if the data is an image.
    std : list or None
        The pixel values of the input data are divided by these
        values, used if the data is an image.
    scale : float
        Specifies the scaling factor to apply to each image.
    model_type : string
        Specifies the deep learning model type (either CNN or RNN).
    max_num_frames : int
        Specifies the maximum number of frames for sequence processing.
       

    Returns
    -------
    string
        String value with SAS deep learning input layer definition

    '''
    config = layer.get_config()
        
    if model_type == 'CNN':
        if (K.image_data_format() == 'channels_first'):
            dummy, C, H, W = config['batch_input_shape']
        else:
            dummy, H, W, C = config['batch_input_shape']
    else:
        if len(config['batch_input_shape']) == 3:
            d1, d2, ts = config['batch_input_shape']
            H = C = 1
            W = ts*max_num_frames
        else:
            return None

    # generate name based on whether layer is actually an input layer
    if (input_layer):
        input_name = config['name']
    else:
        input_name = config['name'] + '_input'

    return write_input_layer(model_name=model_name, layer_name=input_name,
                             channels=str(C), width=str(W),
                             height=str(H), scale=str(scale), 
                             offsets=offsets, std=std,
                             model_type=model_type)


# create SAS residual layer
def keras_residual_layer(layer, model_name, act_func, src_layer):
    '''
    Extract residual layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Add layer
    model_name : string
       Deep learning model name
    act_func : string
       Keras activation function
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       residual layer

    Returns
    -------
    string
        String value with SAS deep learning residual layer definition

    '''
    config = layer.get_config()

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = replace_forward_slash(src_layer[layer.name])
    else:
        raise KerasParseError('Unable to determine source layers for '
                              'residual layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    return write_residual_layer(model_name=model_name, 
                                layer_name=replace_forward_slash(layer.name),
                                activation=layer_act_func, src_layer=source_str)


# create SAS fully connected layer
def keras_full_connect_layer(layer, model_name, act_func, src_layer, layer_dropout, ctc_loss):
    '''
    Extract fully connected layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Fully connected layer
    model_name : string
       Deep learning model name
    act_func : string
       Keras activation function
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       fully connected layer
    layer_dropout : dict
       Dictionary containing dropout layer names (keys) and dropout rates (values)
    ctc_loss : boolean
       Specifies whether to use CTC loss function

    Returns
    -------
    string
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
        source_str = replace_forward_slash(src_layer[layer.name])
    else:
        raise KerasParseError('Unable to determine source layer for '
                              'fully connected layer = ' + layer.name)

    return write_full_connect_layer(model_name=model_name, 
                                    layer_name=replace_forward_slash(layer.name),
                                    nrof_neurons=str(nrof_neurons), nobias=bias_str,
                                    activation=layer_act_func, type=layer_type,
                                    dropout=str(dropout), src_layer=source_str,
                                    ctc_loss=ctc_loss)

# create SAS concatenate layer
def keras_concatenate_layer(layer, model_name, act_func, src_layer):
    '''
    Extract concatenate layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Concatenate layer
    model_name : string
       Deep learning model name
    act_func : string
       Keras activation function
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       residual layer

    Returns
    -------
    string
        String value with SAS deep learning residual layer definition

    '''
    config = layer.get_config()

    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        source_str = replace_forward_slash(src_layer[layer.name])
    else:
        raise KerasParseError('Unable to determine source layers for '
                              'concatenate layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)
    elif ('activation' in config.keys()):
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'identity'

    return write_concatenate_layer(model_name=model_name, 
                                   layer_name=replace_forward_slash(layer.name),
                                   activation=layer_act_func, src_layer=source_str)

# create SAS concatenate layer
def keras_recurrent_layer(layer, model_name, act_func, src_layer):
    '''
    Extract recurrent layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Concatenate layer
    model_name : string
       Deep learning model name
    src_layer : list-of-Layer objects
       Layer objects corresponding to source layer(s) for
       residual layer

    Returns
    -------
    string
        String value with SAS deep learning residual layer definition

    '''
    config = layer.get_config()
        
    # extract source layer(s)
    if layer.name in src_layer.keys():
        source_str = replace_forward_slash(src_layer[layer.name])
    else:
        raise KerasParseError('Unable to determine source layers for '
                              'recurrent layer = ' + layer.name)

    # activation
    if (act_func):
        layer_act_func = map_keras_activation(layer, act_func)    
    elif 'activation' in config.keys():
        layer_act_func = map_keras_activation(layer, config['activation'])
    else:
        layer_act_func = 'auto'
        
    # layer type
    if layer.__class__.__name__.lower() in ['lstm', 'cudnnlstm']:
        rnn_type = 'lstm'
    elif layer.__class__.__name__.lower() in ['gru', 'cudnngru']:
        rnn_type = 'gru'
    else:
        rnn_type = 'rnn'
        
    # sequence output type
    if 'return_sequences' in config.keys():
        if config['return_sequences']:
            seq_output = 'samelength'
        else:
            seq_output = 'encoding'
    else:
        seq_output = 'samelength'
        
    # forward/reverse layer
    if 'go_backwards' in config.keys():
        direction = config['go_backwards']
    else:
        direction = False
        
    # hidden units
    if 'units' in config.keys():
        rnn_size = config['units']
    else:
        raise KerasParseError('Number of hidden neurons not specified '
                              'for layer ' + layer.name)
    
    # bias
    if 'use_bias' in config.keys():
        if not config['use_bias']:
            raise KerasParseError('SAS deep learning requires the use of bias '
                                  'terms.  Cannot import layer ' + layer.name)
                              
    # dropout
    if 'dropout' in config.keys():
        dropout = config['dropout']
    else:
        dropout = 0.0

    return write_recurrent_layer(model_name=model_name, 
                                 layer_name=replace_forward_slash(layer.name),
                                 activation=layer_act_func, src_layer=source_str,
                                 rnn_type=rnn_type, seq_output=seq_output,
                                 direction=direction, rnn_size=rnn_size,
                                 dropout=dropout)
                                   
# extract information from ZeroPadding2D layer
def keras_zeropad2d_layer(layer, src_layer):
    '''
    Extract concatenate layer parameters from layer definition object

    Parameters
    ----------
    layer : Layer object
       Concatenate layer

    Returns
    -------
    zero_pad
        padding 

    '''
    config = layer.get_config()
    zero_pad = {}
    
    # extract source layer(s)
    if (layer.name in src_layer.keys()):
        zero_pad['source_str'] = src_layer[layer.name]
    else:
        raise KerasParseError('Unable to determine source layer for '
                              'zero padding layer = ' + layer.name)
    
    # Keras padding definition:
    #   - If int: the same symmetric padding is applied to height and width.
    #   - If tuple of 2 ints: interpreted as two different symmetric padding values for 
    #       height and width: (symmetric_height_pad, symmetric_width_pad).
    #   - If tuple of 2 tuples of 2 ints: interpreted as 
    #       ((top_pad, bottom_pad), (left_pad, right_pad))

    # determine padding
    padding = config['padding']
    if len(padding) == 1:
        zero_pad['height'] = padding[0]
        zero_pad['width'] = padding[0]
    else:
        if isinstance(padding[0],tuple):
            # height
            if (padding[0][0] == padding[0][1]):
                zero_pad['height'] = padding[0][0]
            else:
                raise DLPyError('Asymmetric padding is not supported')
            # width
            if (padding[1][0] == padding[1][1]):
                zero_pad['width'] = padding[1][0]
            else:
                raise DLPyError('Asymmetric padding is not supported')
        else:
            zero_pad['height'] = padding[0][0]
            zero_pad['width'] = padding[0][1]
    
    return zero_pad
                                   
def map_keras_activation(layer, act_func):
    '''
    Map Keras activation function(s) to SAS activation function(s)

    Parameters
    ----------
    layer : Layer object
       Current layer definition
    act_func : string
       Keras activation function

    Returns
    -------
    string
        SAS activation type

    '''
    class_name = layer.__class__.__name__.lower()
    if class_name in ['conv2d', 'batchnormalization']:
        map_dict = {'softmax': None, 'elu': 'elu', 'selu': None,
                    'softplus': 'softplus', 'softsign': None,
                    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                    'hard_sigmoid': None, 'linear': 'identity'}
    elif class_name == 'dense':
        map_dict = {'softmax': 'softmax', 'elu': 'elu', 'selu': None,
                    'softplus': 'softplus', 'softsign': None,
                    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                    'hard_sigmoid': None, 'linear': 'identity'}
    elif class_name in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
        map_dict = {'softmax': None, 'elu': None, 'selu': None,
                    'softplus': None, 'softsign': None,
                    'relu': 'relu', 'tanh': 'tanh', 'sigmoid': 'sigmoid',
                    'hard_sigmoid': None, 'linear': 'identity'}
    elif class_name == 'add':
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
    Extract activation from layer definition object

    Parameters
    ----------
    layer : Layer object
       Activation layer

    Returns
    -------
    string
        String value with Keras activation function

    '''
    config = layer.get_config()

    return config['activation']


def find_next_computation_layer(model, layer, computation_layer_list):
    '''
    Extract the name of the computation layer following the current layer

    Parameters
    ----------
    model : Model object
       Keras deep learning model
    layer : Layer object
       Current layer object
    computation_layer_list : list
       List of computation layers supported by SAS

    Returns
    -------
    string
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
    Extract the name of the computation layer prior to the current layer

    Parameters
    ----------
    model : Model object
       Keras deep learning model
    layer_name : string
       Current layer name
    computation_layer_list : list
       List of computation layers supported by SAS

    Returns
    -------
    string
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
            try:
                tmp_layer = model.get_layer(name=lname)
                class_name, sublayers = remove_layer_wrapper(tmp_layer)
                prev_layer = tmp_layer
                while (class_name not in computation_layer_list):
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
                    
                    class_name, sublayers = remove_layer_wrapper(tmp_layer)

                for tlayer in sublayers:
                    src_layer_name.append(tlayer.name)
                #src_layer_name.append(tmp_layer.name)        
            except ValueError:
                print("WARNING: could not find layer " + lname + ", in model. Translated model may be inaccurate.")
                src_layer_name.append(lname)

        return src_layer_name


def make_source_str(layer_name):
    '''
    Create a string value representing a Python list of source layers

    Parameters
    ----------
    layer_name : string
       List of Python layer names

    Returns
    -------
    string
        String representation of list of Python layer names

    '''
    source_str = []
    for ii in range(len(layer_name)):
        source_str.append(layer_name[ii])
    return repr(source_str)
