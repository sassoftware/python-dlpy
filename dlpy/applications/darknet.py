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

from dlpy.sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, BN, Pooling, GlobalAveragePooling2D, OutputLayer
from .application_utils import get_layer_options, input_layer_options


def Darknet_Reference(conn, model_table='Darknet_Reference', n_classes=1000, act='leaky',
                      n_channels=3, width=224, height=224, scale=1.0 / 255, random_flip='H',
                      random_crop='UNIQUE', random_mutation=None):

    '''
    Generates a deep learning model with the Darknet_Reference architecture.

    The head of the model except the last convolutional layer is same as
    the head of Tiny Yolov2. Darknet Reference is pre-trained model for
    ImageNet classification.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    act : string
        Specifies the type of the activation function for the batch
        normalization layers and the final convolution layer.
        Default: 'leaky'
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3.
    width : int, optional
        Specifies the width of the input layer.
        Default: 224.
    height : int, optional
        Specifies the height of the input layer.
        Default: 224.
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'h'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'unique'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # conv1 224
    model.add(Conv2d(16, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv4 28
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv5 14
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 7
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=1, pool='max'))
    # conv7 7
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 7
    model.add(Conv2d(1000, width=1, act=act, include_bias=True, stride=1))

    model.add(GlobalAveragePooling2D())
    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def Darknet(conn, model_table='Darknet', n_classes=1000, act='leaky', n_channels=3, width=224, height=224,
            scale=1.0 / 255, random_flip='H', random_crop='UNIQUE', random_mutation=None):
    '''
    Generate a deep learning model with the Darknet architecture.

    The head of the model except the last convolutional layer is
    same as the head of Yolov2. Darknet is pre-trained model for
    ImageNet classification.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 1000
    act : string
        Specifies the type of the activation function for the batch
        normalization layers and the final convolution layer.
        Default: 'leaky'
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the
        input layer.
        Default: 3.
    width : int, optional
        Specifies the width of the input layer.
        Default: 224.
    height : int, optional
        Specifies the height of the input layer.
        Default: 224.
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel
        intensity values.
        Default: 1.0 / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'h'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'unique'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))  # route
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv19 7 13
    model.add(Conv2d(1000, width=1, act=act, include_bias=True, stride=1))
    # model.add(BN(act = actx))

    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model