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

import sys


# model/input layer definition
def write_input_layer(model_name='sas', layer_name='data', channels='-1',
                      width='-1', height='-1', scale='1.0'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning input layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    channels : [string]
       number of input channels
    width : [string]
       image width
    height : [string]
       image height
    scale : [string]
       scaling factor to apply to raw image pixel data

    Returns
    -------
    String representing Python code defining a SAS deep learning input layer
    '''

    code_string = ("import sys \n"
                   "\n"
                   "def " + model_name.lower() +
                    "_model(s, input_crop_type=None, input_channel_offset=None, input_image_size=None):\n"
                    "\n"
                    "   # quick error-checking and default setting \n"
                    "   if (input_crop_type is None): \n"
                    "       input_crop_type='NONE' \n"
                    "   else: \n"
                    "       if (input_crop_type.upper() != 'NONE') and (input_crop_type.upper() != 'UNIQUE'): \n"
                    "           sys.exit('ERROR: input_crop_type can only be NONE or UNIQUE') \n"
                    " \n"
                    "   if (input_channel_offset is None): \n"
                    "       print('INFO: setting channel mean values to ImageNet means') \n"
                    "       input_channel_offset = [103.939, 116.779, 123.68] \n"
                    " \n"
                    "   if (input_image_size is not None): \n"
                    "       channels = input_image_size[0] \n"
                    "       if (len(input_image_size) == 2): \n"
                    "           height = width = input_image_size[1] \n"
                    "       elif (len(inputImageSize) == 3): \n"
                    "           height,width = input_image_size[1:] \n"
                    "       else: \n"
                    "           sys.exit('ERROR: input_image_size must be a tuple with two or three entries') \n"
                    " \n"
                    "   # instantiate model \n"
                    "   s.buildModel(model=dict(name='" + model_name + "',replace=True),type='CNN') \n"
                    " \n"
                    "   # input layer \n"
                    "   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                    "              layer=dict( type='input', nchannels=" + channels + ", width=" + width + ", height=" + height + ", \n"
                    "              scale = " + scale + ", randomcrop=input_crop_type, offsets=input_channel_offset))")

    return code_string


# convolution layer definition
def write_convolution_layer(model_name='sas', layer_name='conv', nfilters='-1',
                            width='3', height='3', stride='1', nobias='False',
                            activation='identity', dropout='0', src_layer='none'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning convolution layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    nfilters : [string]
       number of output feature maps
    width : [string]
       image width
    height : [string]
       image height
    stride : [string]
       vertical/horizontal step size in pixels
    nobias : [string]
       omit (True) or retain (False) the bias term
    activation : [string]
       activation function
    dropout : [string]
       dropout factor (0 < dropout < 1.0)
    src_layer : [string]
       source layer(s) for the convolution layer

    Returns
    -------
    String representing Python code defining a SAS deep learning convolution layer
    '''

    # create program string
    code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                   "              layer=dict(type='convolution', nfilters=" + nfilters + ", width=" + width + ", height=" + height + ",\n"
                   "                         stride=" + stride + ", nobias=" + nobias + ", act='" + activation + "', dropout=" + dropout + "), \n"
                   "              srcLayers=" + src_layer + ")")

    return code_string


# batch normalization layer definition
def write_batch_norm_layer(model_name='sas', layer_name='bn',
                           activation='identity', src_layer='none'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning batch normalization layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    activation : [string]
       activation function
    src_layer : [string]
       source layer(s) for the convolution layer

    Returns
    -------
    String representing Python code defining a SAS deep learning batch normalization layer
    '''

    code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                   "              layer=dict( type='batchnorm', act='" + activation + "'), \n"
                   "              srcLayers=" + src_layer + ")")

    return code_string


# pooling layer definition
def write_pooling_layer(model_name='sas', layer_name='pool',
                        width='2', height='2', stride='2', type='max',
                        dropout='0', src_layer='none'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning pooling layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    width : [string]
       image width
    height : [string]
       image height
    stride : [string]
       vertical/horizontal step size in pixels
    type : [string]
       pooling type
    dropout : [string]
       dropout factor (0 < dropout < 1.0)
    src_layer : [string]
       source layer(s) for the convolution layer

    Returns
    -------
    String representing Python code defining a SAS deep learning pooling layer
    '''

    # create program string
    code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                   "              layer=dict(type='pooling',width=" + width + ", height=" + height + ", \n"
                   "                         stride=" + stride + ", pool='" + type + "', dropout=" + dropout + "), \n"
                   "              srcLayers=" + src_layer + ")")

    return code_string


# residual layer definition
def write_residual_layer(model_name='sas', layer_name='residual',
                         activation='identity', src_layer='none'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning residual layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    activation : [string]
       activation function
    src_layer : [string]
       source layer(s) for the convolution layer

    Returns
    -------
    String representing Python code defining a SAS deep learning residual layer
    '''

    # create program string
    code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                   "              layer=dict( type='residual', act='" + activation + "'), \n"
                   "              srcLayers=" + src_layer + ")")

    return code_string


# fully connected layer definition
def write_full_connect_layer(model_name='sas', layer_name='fullconnect',
                             nrof_neurons='-1', nobias='true',
                             activation='identity', type='fullconnect', dropout='0',
                             src_layer='none'):
    '''
    Function to generate a string representing Python code defining a
    SAS deep learning fully connected layer

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model
    layer_name : [string]
       Layer name
    nrof_neurons : [string]
       number of output neurons
    nobias : [string]
       omit (True) or retain (False) the bias term
    activation : [string]
       activation function
    type : [string]
       fully connected layer type (fullconnect or output)
    dropout : [string]
       dropout factor (0 < dropout < 1.0)
    src_layer : [string]
       source layer(s) for the convolution layer

    Returns
    -------
    String representing Python code defining a SAS deep learning convolution layer
    '''

    if (type == 'fullconnect'):
        code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                       "              layer=dict(type='" + type + "', n=" + nrof_neurons + ",\n"
                         "                         nobias=" + nobias + ", act='" + activation + "', dropout=" + dropout + "), \n"
                                                        "              srcLayers=" + src_layer + ")")
    else:
        code_string = ("   s.addLayer(model='" + model_name + "', name='" + layer_name + "', \n"
                       "              layer=dict(type='" + type + "', n=" + nrof_neurons + ",\n"
                         "                         nobias=" + nobias + ", act='" + activation + "'), \n"
                              "              srcLayers=" + src_layer + ")")

    return code_string


# Python __main__ function
def write_main_entry(model_name):
    '''
    Function to generate a string representing Python code defining
    the __main__ Python entry point

    Parameters:

    ----------
    model_name : [string]
       Name for deep learning model

    Returns
    -------
    String representing Python code defining the __main__ entry point
    '''

    # create program string
    code_string = ("############################################################# \n"
                   "if __name__ == '__main__': \n"
                   "   sys.exit('ERROR: this module defines the " + model_name + " model.  Do not call directly.') \n")

    return code_string


#############################################################
if __name__ == '__main__':
    sys.exit('ERROR: this module cannot be invoked from the command line.')
