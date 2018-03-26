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

''' Write python codes for create sas model '''

import sys


# model/input layer definition
def write_input_layer(model_name='sas', layer_name='data', channels='-1',
                      width='-1', height='-1', scale='1.0'):
    '''
    Generate Python code defining a SAS deep learning input layer

    Parameters
    ----------
    model_name : string
       Name for deep learning model
    layer_name : string
       Layer name
    channels : string
       number of input channels
    width : string
       image width
    height : string
       image height
    scale : string
       scaling factor to apply to raw image pixel data

    Returns
    -------
    string
        String representing Python code defining a SAS deep learning input layer

    '''
    out = [
        'import sys',
        '',
        'def sas_model_gen(s, input_crop_type=None, input_channel_offset=None, input_image_size=None):',
        '   # quick check for deeplearn actionset',
        '   actionset_list = s.actionsetinfo().setinfo.actionset.tolist()',
        '   actionset_list = [item.lower() for item in actionset_list]',
        '   if "deeplearn" not in actionset_list:s.loadactionset("deeplearn")',
        '   ',
        '   # quick error-checking and default setting',
        '   if (input_crop_type is None):',
        '       input_crop_type="NONE"',
        '   else:',
        '       if (input_crop_type.upper() != "NONE") and (input_crop_type.upper() != "UNIQUE"):',
        '           sys.exit("ERROR: input_crop_type can only be NONE or UNIQUE")',
        '',
        '   if (input_image_size is not None):',
        '       channels = input_image_size[0]',
        '       if (len(input_image_size) == 2):',
        '           height = width = input_image_size[1]',
        '       elif (len(inputImageSize) == 3):',
        '           height,width = input_image_size[1:]',
        '       else:',
        '           sys.exit("ERROR: input_image_size must be a tuple with two or three entries")',
        '',
        '   # instantiate model',
        '   s.buildModel(model=dict(name=' + repr(model_name) + ',replace=True),type="CNN")',
        '',
        '   # input layer',
        '   nchannels=' + channels,
        '   if input_channel_offset is None and nchannels==3:',
        '       print("INFO: setting channel mean values to ImageNet means")',
        '       input_channel_offset = [103.939, 116.779, 123.68]',
        '       s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '                  layer=dict( type="input", nchannels=' + channels + ', width=' + width + ', height=' + height + ',',
        '                  scale = ' + scale + ', randomcrop=input_crop_type, offsets=input_channel_offset))',
        '   elif input_channel_offset is not None:',
        '       s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '                  layer=dict( type="input", nchannels=' + channels + ', width=' + width + ', height=' + height + ',',
        '                  scale = ' + scale + ', randomcrop=input_crop_type, offsets=input_channel_offset))',
        '   else:',
        '       s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '                  layer=dict( type="input", nchannels=' + channels + ', width=' + width + ', height=' + height + ',',
        '                  scale = ' + scale + ', randomcrop=input_crop_type))'
    ]
    return '\n'.join(out)


# convolution layer definition
def write_convolution_layer(model_name='sas', layer_name='conv', nfilters='-1',
                            width='3', height='3', stride='1', nobias='False',
                            activation='identity', dropout='0', src_layer='none'):
    '''
    Generate Python code defining a SAS deep learning convolution layer

    Parameters
    ----------
    model_name : string, optional
       Name for deep learning model
    layer_name : string, optional
       Layer name
    nfilters : string, optional
       number of output feature maps
    width : string, optional
       image width
    height : string, optional
       image height
    stride : string, optional
       vertical/horizontal step size in pixels
    nobias : string, optional
       omit (True) or retain (False) the bias term
    activation : string, optional
       activation function
    dropout : string, optional
       dropout factor (0 < dropout < 1.0)
    src_layer : string, optional
       source layer(s) for the convolution layer

    Returns
    -------
    string

    '''
    out = [
        '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '              layer=dict(type="convolution", nfilters=' + nfilters + ', width=' + width + ', height=' + height + ',',
        '                         stride=' + stride + ', nobias=' + nobias + ', act=' + repr(
            activation) + ', dropout=' + dropout + '), \n',
        '              srcLayers=' + src_layer + ')'
    ]
    return '\n'.join(out)


# batch normalization layer definition
def write_batch_norm_layer(model_name='sas', layer_name='bn',
                           activation='identity', src_layer='none'):
    '''
    Generate Python code defining a SAS deep learning batch normalization layer

    Parameters
    ----------
    model_name : string, optional
       Name for deep learning model
    layer_name : string, optional
       Layer name
    activation : string, optional
       activation function
    src_layer : string, optional
       source layer(s) for the convolution layer

    Returns
    -------
    string

    '''
    out = [
        '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '              layer=dict( type="batchnorm", act=' + repr(activation) + '),',
        '              srcLayers=' + src_layer + ')'
    ]
    return '\n'.join(out)


# pooling layer definition
def write_pooling_layer(model_name='sas', layer_name='pool',
                        width='2', height='2', stride='2', type='max',
                        dropout='0', src_layer='none'):
    '''
    Generate Python code defining a SAS deep learning pooling layer

    Parameters
    ----------
    model_name : string, optional
       Name for deep learning model
    layer_name : string, optional
       Layer name
    width : string, optional
       image width
    height : string, optional
       image height
    stride : string, optional
       vertical/horizontal step size in pixels
    type : string, optional
       pooling type
    dropout : string, optional
       dropout factor (0 < dropout < 1.0)
    src_layer : string, optional
       source layer(s) for the convolution layer

    Returns
    -------
    string

    '''
    out = [
        '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '              layer=dict(type="pooling", width=' + width + ', height=' + height + ',',
        '                         stride=' + stride + ', pool=' + repr(type) + ', dropout=' + dropout + '),',
        '              srcLayers=' + src_layer + ')'
    ]
    return '\n'.join(out)


# residual layer definition
def write_residual_layer(model_name='sas', layer_name='residual',
                         activation='identity', src_layer='none'):
    '''
    Generate Python code defining a SAS deep learning residual layer

    Parameters
    ----------
    model_name : string, optional
       Name for deep learning model
    layer_name : string, optional
       Layer name
    activation : string, optional
       activation function
    src_layer : string, optional
       source layer(s) for the convolution layer

    Returns
    -------
    string

    '''
    out = [
        '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
        '              layer=dict( type="residual", act="' + activation + '"),',
        '              srcLayers=' + src_layer + ')'
    ]
    return '\n'.join(out)


# fully connected layer definition
def write_full_connect_layer(model_name='sas', layer_name='fullconnect',
                             nrof_neurons='-1', nobias='true',
                             activation='identity', type='fullconnect', dropout='0',
                             src_layer='none'):
    '''
    Generate Python code defining a SAS deep learning fully connected layer

    Parameters
    ----------
    model_name : string, optional
       Name for deep learning model
    layer_name : string, optional
       Layer name
    nrof_neurons : string, optional
       number of output neurons
    nobias : string, optional
       omit (True) or retain (False) the bias term
    activation : string, optional
       activation function
    type : string, optional
       fully connected layer type (fullconnect or output)
    dropout : string, optional
       dropout factor (0 < dropout < 1.0)
    src_layer : string, optional
       source layer(s) for the convolution layer

    Returns
    -------
    string

    '''
    if (type == 'fullconnect'):
        out = [
            '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
            '              layer=dict(type=' + repr(type) + ', n=' + nrof_neurons + ',',
            '                         nobias=' + nobias + ', act=' + repr(activation) + ', dropout=' + dropout + '),',
            '              srcLayers=' + src_layer + ')'
        ]
    else:
        out = [
            '   s.addLayer(model=' + repr(model_name) + ', name=' + repr(layer_name) + ',',
            '              layer=dict(type=' + repr(type) + ', n=' + nrof_neurons + ',',
            '                         nobias=' + nobias + ', act=' + repr(activation) + '),',
            '              srcLayers=' + src_layer + ')'
        ]
    return '\n'.join(out)


# Python __main__ function
def write_main_entry(model_name):
    '''
    Generate Python code defining the __main__ Python entry point

    Parameters
    ----------
    model_name : string
       Name for deep learning model

    Returns
    -------
    string

    '''
    out = [
        '#############################################################',
        'if __name__ == "__main__":',
        '   sys.exit("ERROR: this module defines the ' + model_name + ' model.  Do not call directly.")'
    ]
    return '\n'.join(out)


#############################################################
if __name__ == '__main__':
    sys.exit('ERROR: this module cannot be invoked from the command line.')
