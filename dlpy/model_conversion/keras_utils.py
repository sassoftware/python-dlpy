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

import keras
from keras.layers import LSTM, Bidirectional, GRU
from dlpy.utils import DLPyError

''' Keras-specific utilities '''

def remove_layer_wrapper(layer):
    '''
    Determines underlying layer type for wrapped layers

    Parameters
    ----------
    layer : Layer object
       Current layer object

    Returns
    -------
    string
        class name of wrapped layer
    list of layer objects
        unwrapped layer object(s)

    '''
    class_name = layer.__class__.__name__.lower()
    # check for layer wrappers
    sublayers = []
    if class_name == 'timedistributed':
        layer_info = layer.get_config()['layer']
        layer_info['config']['name'] = layer.name
        class_name = layer_info['class_name'].lower()
        if class_name == 'dense':
            sublayers.append(keras.layers.Dense(**layer_info['config']))
        else:
            raise DLPyError(class_name + ' is an unsupported time distributed '
                            'layer type - model conversion failed')
    elif class_name == 'bidirectional':
        layer_info = layer.get_config()['layer']
        class_name = layer_info['class_name'].lower()
        # forward direction
        layer_info['config']['name'] = layer.forward_layer.name
        layer_info['config']['go_backwards'] = False
        if class_name == 'lstm':
            sublayers.append(keras.layers.LSTM(**layer_info['config']))
        elif class_name == 'gru':
            sublayers.append(keras.layers.GRU(**layer_info['config']))
        elif class_name == 'simplernn':
            sublayers.append(keras.layers.SimpleRNN(**layer_info['config']))
        elif class_name == 'cudnnlstm':
            sublayers.append(keras.layers.CuDNNLSTM(**layer_info['config']))
        elif class_name == 'cudnngru':
            sublayers.append(keras.layers.CuDNNGRU(**layer_info['config']))
        else:
            raise DLPyError(class_name + ' is an unsupported time distributed '
                            'layer type - model conversion failed')
        # backward direction
        layer_info['config']['name'] = layer.backward_layer.name
        layer_info['config']['go_backwards'] = True
        if class_name == 'lstm':
            sublayers.append(keras.layers.LSTM(**layer_info['config']))
        elif class_name == 'gru':
            sublayers.append(keras.layers.GRU(**layer_info['config']))
        elif class_name == 'simplernn':
            sublayers.append(keras.layers.SimpleRNN(**layer_info['config']))
        elif class_name == 'cudnnlstm':
            sublayers.append(keras.layers.CuDNNLSTM(**layer_info['config']))
        elif class_name == 'cudnngru':
            sublayers.append(keras.layers.CuDNNGRU(**layer_info['config']))
        else:
            raise DLPyError(class_name + ' is an unsupported time distributed '
                            'layer type - model conversion failed')
    else:
        sublayers.append(layer)
        
    # Must return sublayers in reverse order if CUDNN is used.
    # This aligns the Viya layer mapping with the CUDNN layer
    # mapping.
    if layer.__class__.__name__.lower() == 'bidirectional':
        sublayer_info = layer.get_config()['layer']
        if sublayer_info['class_name'].lower() in ['cudnnlstm','cudnngru']:
            sublayers.reverse()
            #sublayers = [sublayers[1], sublayers[0]]

    return class_name, sublayers
    
def create_cpu_compatible_layer(layer, model_type='CNN'):
    '''
    Creates a new layer object using parameters from the 
    provided layer

    Parameters
    ----------
    layer : Layer object
       Current layer object
    model_type : string, optional
       Current model type (one of 'CNN' or 'RNN')

    Returns
    -------
    Layer object

    '''
    
    if model_type == 'RNN':    
        # check for the use of CUDNN RNN layers
        # these layers must be mapped to non-CUDNN layer
        # format
        if layer.__class__.__name__ == 'Bidirectional':
            tlayer = layer.forward_layer
            config = tlayer.get_config()
            if tlayer.__class__.__name__ == 'CuDNNLSTM':
                new_layer = Bidirectional(LSTM(config['units'], 
                                               return_sequences=config['return_sequences'],
                                               return_state=False,
                                               unit_forget_bias=config['unit_forget_bias'],
                                               stateful=False,
                                               activation='tanh',
                                               recurrent_activation='sigmoid'), merge_mode='concat')
            elif tlayer.__class__.__name__ == 'CuDNNGRU':
                new_layer = Bidirectional(GRU(config['units'], 
                                              return_sequences=config['return_sequences'], 
                                              return_state=False, 
                                              stateful=False, 
                                              reset_after=True), merge_mode='concat')
            else:
                new_layer = layer
        else:
            tlayer = layer
            config = tlayer.get_config()
            if tlayer.__class__.__name__ == 'CuDNNLSTM':
                new_layer = LSTM(config['units'], 
                                 return_sequences=config['return_sequences'],
                                 return_state=False,
                                 unit_forget_bias=config['unit_forget_bias'],
                                 stateful=False,
                                 activation='tanh',
                                 recurrent_activation='sigmoid')
            elif tlayer.__class__.__name__ == 'CuDNNGRU':
                new_layer = GRU(config['units'], 
                                return_sequences=config['return_sequences'], 
                                return_state=False, 
                                stateful=False, 
                                reset_after=True)
            else:
                new_layer = layer
    else:
        new_layer = layer
        
    return new_layer
