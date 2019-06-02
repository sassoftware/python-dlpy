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

''' Model conversion utilities '''

def replace_forward_slash(layer_name):
    '''
    Replaces forward slash (/) in layer names with _

    Parameters
    ----------
    layer_name : string
       Layer name

    Returns
    -------
    string
        Layer name with / replaced with _

    '''
    return layer_name.replace('/','_')

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

def query_action_parm(conn, action_name, action_set, parm_name):
    '''
    Check whether action includes given parameter

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    action_name : string
        The name of the action
    action_set : string
        The name of the action set that contains the action
    parm_name : string
        The parameter name.

    Returns
    -------
    boolean
        Indicates whether action supports parameter
    list of dictionaries
        Dictionaries that describe action parameters
        
    '''
    
    # check whether action set is loaded
    parm_valid = False
    act_parms = []
    r = conn.retrieve('queryactionset', _messagelevel='error', actionset=action_set)
    if r[action_set]:
        # check whether action part of action set
        r = conn.retrieve('listactions', _messagelevel='error', actionset=action_set)
        if action_name in r[action_set]['name'].tolist():
            r = conn.retrieve('builtins.reflect', action=action_name,
                              actionset=action_set)
    
            # check for parameter
            act_parms = r[0]['actions'][0]['params']
            for pdict in act_parms:
                if pdict['name'].lower() == parm_name.lower():
                    parm_valid = True
                    break
        else:
            raise DLPyError(action_name + ' is not an action in the '
                            + action_set + ' action set.')
    else:
        raise DLPyError(action_set + ' is not valid or not currently loaded.')
        
    return parm_valid, act_parms

def check_rnn_import(conn):
    '''
    Check whether importing RNN models is supported
    
    Parameters
    ----------
    conn : CAS
        The CAS connection object

    Returns
    -------
    boolean
        Indicates whether importing RNN models is supported

    '''
    
    rnn_valid, act_parms = query_action_parm(conn, 'dlImportModelWeights', 'deepLearn', 'gpuModel')
        
    return rnn_valid
    
def check_normstd(conn):
    '''
    Check whether normStd option for addLayer action supported
    
    Parameters
    ----------
    conn : CAS
        The CAS connection object

    Returns
    -------
    boolean
        Indicates whether normStd option is supported

    '''
    
    dummy, act_parms = query_action_parm(conn, 'addLayer', 'deepLearn', 'layer')
    norm_std = False
    for pdict in act_parms:
        if pdict['name'] == 'layer':
            for tmp_dict in pdict['alternatives'][0]['parmList']:
                if tmp_dict['name'].lower() == 'normstds':
                    norm_std = True
                    break
        
    return norm_std
    