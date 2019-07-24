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
    