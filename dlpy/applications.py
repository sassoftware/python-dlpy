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

''' Pre-built deep learning models '''

import os
import warnings

from .Sequential import Sequential
from .blocks import ResBlockBN, ResBlock_Caffe, DenseNetBlock
from .caffe_models import (model_vgg16, model_vgg19, model_resnet50,
                           model_resnet101, model_resnet152)
from .layers import (InputLayer, Conv2d, Pooling, Dense, BN, OutputLayer)
from .model import Model
from .utils import random_name


def LeNet5(conn, model_table='LENET5',
           n_classes=10, n_channels=1, width=28, height=28, scale=1.0 / 255,
           random_flip='none', random_crop='none', offsets=0):
    '''
    Generate a deep learning model with LeNet5 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 1
    width : int, optional
        Specifies the width of the input layer
        Default: 28
    height : int, optional
        Specifies the height of the input layer
        Default: 28
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 10
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data
        is used. Images are cropped to the values that are specified in the
        width and height parameters. Only the images with one or both
        dimensions that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: 0

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=6, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=16, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=120))
    model.add(Dense(n=84))
    model.add(OutputLayer(n=n_classes))

    return model


def LeNet5_bn(conn, model_table='LENET_BN',
              n_channels=1, width=28, height=28, n_classes=10, scale=1.0 / 255,
              random_flip='none', random_crop='none', offsets=0):
    '''
    Generate a LeNet Model with Batch normalization

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 1
    width : int, optional
        Specifies the width of the input layer
        Default: 28
    height : int, optional
        Specifies the height of the input layer
        Default: 28
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 10
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1/255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: 0

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop, name='mnist'))

    model.add(Conv2d(n_filters=20, width=5, act='identity', stride=1,
                     includeBias=False, name='conv1'))
    model.add(BN(act='relu', name='conv1_bn'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max', name='pool1'))

    model.add(Conv2d(n_filters=50, width=5, act='identity', stride=1,
                     includeBias=False, name='conv2'))
    model.add(BN(act='relu', name='conv2_bn'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max', name='pool2'))

    model.add(Dense(n=500, name='ip1'))
    model.add(OutputLayer(n=n_classes, name='ip2'))

    return model


def VGG11(conn, model_table='VGG11',
          n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with VGG11 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if offsets is None:
        offsets = (103.939, 116.779, 123.68)

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def VGG11_bn(conn, model_table='VGG11',
             n_classes=1000, n_channels=3, width=224, height=224, scale=1,
             random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generate deep learning model with VGG11 architecture with batch normalization layers

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default: 'VGG11_BN'
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data
        is used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data
        is used. Images are cropped to the values that are specified in the
        width and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if offsets is None:
        offsets = (103.939, 116.779, 123.68)

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity',
                     includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def VGG13(conn, model_table='VGG13',
          n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with VGG13 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default	: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if offsets is None:
        offsets = (103.939, 116.779, 123.68)

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def VGG13_bn(conn, model_table='VGG13',
             n_classes=1000, n_channels=3, width=224, height=224, scale=1,
             random_flip='none', random_crop='none', offsets=None):
    '''
    Generate deep learning model with VGG13 architecture with batch normalization layers

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: None
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of--doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if offsets is None:
        offsets = (103.939, 116.779, 123.68)

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def VGG16(conn, model_table='VGG16',
          n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_train_weight=False, pre_train_weight_file=None, include_top=False):
    '''
    Generate a deep learning model with VGG16 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_train_weight : boolean, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_train_weight_file : string, required when pre_train_weight=True.
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
    include_top : boolean, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the FC layers
        Default: False

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_train_weight:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))

        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Dense(n=4096, dropout=0.5))
        model.add(Dense(n=4096, dropout=0.5))
        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_train_weight_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_train_weight_file using the fully qualified server side path.')
        model_cas = model_vgg16.VGG16_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            label_table = random_name('label')
            label_file = os.path.join(os.path.dirname(__file__),
                                      'datasources', 'imagenet_label.sas7bdat')
            conn.upload(
                casout=dict(replace=True, name=label_table),
                data=label_file)

            model.load_weights(path=pre_train_weight_file,
                               labeltable=dict(name=label_table,
                                               varS=['levid', 'levname']))
            model._retrieve_('table.droptable', table=label_table)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_train_weight_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<19'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def VGG16_bn(conn, model_table='VGG16',
             n_classes=1000, n_channels=3, width=224, height=224, scale=1,
             random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generate deep learning model with VGG16 architecture with batch normalization layers

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: None
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def VGG19(conn, model_table='VGG19',
          n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_train_weight=False, pre_train_weight_file=None, include_top=False):
    '''
    Generate a deep learning model with VGG19 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_train_weight : boolean, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_train_weight_file : string, required when pre_train_weight=True.
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
    include_top : boolean, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the FC layers.
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is False
    :class:`Model`
        If `pre_train_teight` is True

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_train_weight:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))

        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
        model.add(Pooling(width=2, height=2, stride=2, pool='max'))

        model.add(Dense(n=4096, dropout=0.5))
        model.add(Dense(n=4096, dropout=0.5))
        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_train_weight_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_train_weight_file using the fully qualified server side path.')
        model_cas = model_vgg19.VGG19_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            label_table = random_name('label')
            label_file = os.path.join(os.path.dirname(__file__),
                                      'datasources', 'imagenet_label.sas7bdat')
            conn.upload(
                casout=dict(replace=True, name=label_table),
                data=label_file)

            model.load_weights(path=pre_train_weight_file,
                               labeltable=dict(name=label_table,
                                               varS=['levid', 'levname']))
            model._retrieve_('table.droptable', table=label_table)
            return model

        else:

            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_train_weight_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<22'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def VGG19_bn(conn, model_table='VGG19',
             n_classes=1000, n_channels=3, width=224, height=224, scale=1,
             random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generate deep learning model with VGG19 architecture with batch normalization layers

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string
        Specifies the name of CAS table to store the model in
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default	: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1,
                     act='identity', includeBias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model


def ResNet18_SAS(conn, model_table='RESNET18_SAS', batch_norm_first=True,
                 n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 random_flip='none', random_crop='none',
                 offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with ResNet18 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, optional
        Specifies the name of CAS table to store the model
        Default: 'RESNET18_SAS'
    batch_norm_first: boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default	: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [2, 2, 2, 2]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet18_Caffe(conn, model_table='RESNET18_CAFFE', batch_norm_first=False,
                   n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   random_flip='none', random_crop='none', offsets=None):
    '''
    Generate a deep learning model with ResNet18 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [2, 2, 2, 2]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if rep_num == 0:
                conv_short_cut = True
                if i == 0:
                    strides = 1
                else:
                    strides = 2
            else:
                conv_short_cut = False
                strides = 1
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                     n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first,
                                     conv_short_cut=conv_short_cut))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34_SAS(conn, model_table='RESNET34_SAS', batch_norm_first=True,
                 n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 random_flip='none', random_crop='none',
                 offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with ResNet34 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first: boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34_Caffe(conn, model_table='RESNET34_CAFFE', batch_norm_first=False,
                   n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   random_flip='none', random_crop='none', offsets=None):
    '''
    Generate deep learning model with ResNet34 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''

    # conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')
    # model_resnet18.ResNet18_Model(s=conn, model_table=model_table,
    #                               n_classes=n_classes, random_crop=random_crop,
    #                               random_flip=random_flip, offsets=offsets)
    # model = Model.from_table(conn.CASTable(model_table))
    # return model
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    # Configuration of the residual blocks
    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if rep_num == 0:
                conv_short_cut = True
                if i == 0:
                    strides = 1
                else:
                    strides = 2
            else:
                conv_short_cut = False
                strides = 1
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                     n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first,
                                     conv_short_cut=conv_short_cut))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_SAS(conn, model_table='RESNET50_SAS', batch_norm_first=True,
                 n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 random_flip='none', random_crop='none',
                 offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with ResNet50 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or list-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_Caffe(conn, model_table='RESNET50_CAFFE', batch_norm_first=False,
                   n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   random_flip='none', random_crop='none',
                   offsets=(103.939, 116.779, 123.68),
                   pre_train_weight=False, pre_train_weight_file=None, include_top=False):
    '''
    Generate deep learning model with ResNet50 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_train_weight : boolean, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_train_weight_file : string, required when pre_train_weight=True.
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
    include_top : boolean, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_train_weight:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512),
                          (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 4, 6, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_train_weight_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_train_weight_file using the fully qualified server side path.')
        model_cas = model_resnet50.ResNet50_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            label_table = random_name('label')
            label_file = os.path.join(os.path.dirname(__file__),
                                      'datasources', 'imagenet_label.sas7bdat')
            conn.upload(
                casout=dict(replace=True, name=label_table),
                data=label_file)

            model.load_weights(path=pre_train_weight_file,
                               labeltable=dict(name=label_table,
                                               varS=['levid', 'levname']))
            model._retrieve_('table.droptable', table=label_table)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_train_weight_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<125'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet101_SAS(conn, model_table='RESNET101_SAS', batch_norm_first=True,
                  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  random_flip='none', random_crop='none',
                  offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with ResNet101 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data
        is used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default	: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 23, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet101_Caffe(conn, model_table='RESNET101_CAFFE', batch_norm_first=False,
                    n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    random_flip='none', random_crop='none',
                    offsets=(103.939, 116.779, 123.68),
                    pre_train_weight=False, pre_train_weight_file=None, include_top=False):
    '''
    Generate deep learning model with ResNet101 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_train_weight : boolean, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_train_weight_file : string, required when pre_train_weight=True.
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
    include_top : boolean, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False.

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_train_weight:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512),
                          (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 4, 23, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_train_weight_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_train_weight_file using the fully qualified server side path.')
        model_cas = model_resnet101.ResNet101_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            label_table = random_name('label')
            label_file = os.path.join(os.path.dirname(__file__),
                                      'datasources', 'imagenet_label.sas7bdat')
            conn.upload(
                casout=dict(replace=True, name=label_table),
                data=label_file)

            model.load_weights(path=pre_train_weight_file,
                               labeltable=dict(name=label_table,
                                               varS=['levid', 'levname']))
            model._retrieve_('table.droptable', table=label_table)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_train_weight_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<244'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet152_SAS(conn, model_table='RESNET152_SAS', batch_norm_first=True,
                  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  random_flip='none', random_crop='none',
                  offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with ResNet152 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 8, 36, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                    height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet152_Caffe(conn, model_table='RESNET152_CAFFE', batch_norm_first=False,
                    n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    random_flip='none', random_crop='none',
                    offsets=(103.939, 116.779, 123.68),
                    pre_train_weight=False, pre_train_weight_file=None, include_top=False):
    '''
    Generate deep learning model with ResNet152 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'none'
    offsets : double, or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_train_weight : boolean, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_train_weight_file : string, required when pre_train_weight=True.
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
    include_top : boolean, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default : False

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_train_weight:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512),
                          (256, 256, 1024), (512, 512, 2048)]
        rep_nums_list = [3, 8, 36, 3]

        for i in range(4):
            kernel_sizes = kernel_sizes_list[i]
            n_filters = n_filters_list[i]
            for rep_num in range(rep_nums_list[i]):
                if rep_num == 0:
                    conv_short_cut = True
                    if i == 0:
                        strides = 1
                    else:
                        strides = 2
                else:
                    conv_short_cut = False
                    strides = 1
                model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes,
                                         n_filters=n_filters, strides=strides,
                                         batch_norm_first=batch_norm_first,
                                         conv_short_cut=conv_short_cut))

        # Bottom Layers
        pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                        height // 2 // 2 // 2 // 2 // 2)
        model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model
    else:
        if pre_train_weight_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_train_weight_file using the fully qualified server side path.')
        model_cas = model_resnet152.ResNet152_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            label_table = random_name('label')
            label_file = os.path.join(os.path.dirname(__file__),
                                      'datasources', 'imagenet_label.sas7bdat')
            conn.upload(
                casout=dict(replace=True, name=label_table),
                data=label_file)

            model.load_weights(path=pre_train_weight_file,
                               labeltable=dict(name=label_table,
                                               varS=['levid', 'levname']))
            model._retrieve_('table.droptable', table=label_table)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_train_weight_file)
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc1000')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='output',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<363'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def wide_resnet(conn, model_table='WIDE_RESNET', batch_norm_first=True, depth=2,
                k=4, n_classes=None, n_channels=3, width=32, height=32, scale=1,
                random_flip='none', random_crop='none', offsets=(114, 122, 125)):
    '''
    Generate a deep learning model with ResNet152 architecture

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : boolean, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    depth : int
        Specifies the number of convolution layers added into the model
        Default: 2
    k : int
        Specifies the widening factor
        Default: 4
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    scale : double, optional
        Specifies a scaling factor to apply to each image
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'none', 'v'
        Default: 'hv'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    n_stack = int((depth - 2) / 6)
    in_filters = 16

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(in_filters, 3, act='identity', includeBias=False, stride=1))
    model.add(BN(act='relu'))

    # Residual block configuration.
    n_filters_list = [(16 * k, 16 * k), (32 * k, 32 * k), (64 * k, 64 * k)]
    kernel_sizes_list = [(3, 3)] * len(n_filters_list)
    rep_nums_list = [n_stack, n_stack, n_stack]

    for i in range(len(n_filters_list)):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2, height // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def DenseNet_Cifar(conn, model_table='DenseNet_Cifar', n_classes=None, conv_channel=16, growth_rate=12,
                   n_blocks=4, n_cells=4, n_channels=3, width=32, height=32, scale=1,
                   random_flip='none', random_crop='none', offsets=(85, 111, 139)):
    '''
    Generate a deep learning model with DenseNet architecture

    Parameters
    ----------
    conn :
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional.
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: None
    conv_channel: int, optional.
        Specifies the number of filters of first convolution layer.
        Default : 16
    growth_rate: int, optional.
        Specifies growth rate of convolution layer.
        Default : 12
    n_blocks : int, optional.
        Specifies the number of DenseNetBlocks.
        Default : 4
    n_cells : int, optional.
        Specifies the number of densely connection in each DenseNetBlock
        Default : 4
    n_channels : double, optional.
        Specifies the number of the channels of the input layer.
        Default : 3.
    width : double, optional.
        Specifies the width of the input layer.
        Default : 224.
    height : double, optional.
        Specifies the height of the input layer.
        Default : 224.
    scale : double, optional.
        Specifies a scaling factor to apply to each image..
        Default : 1.
    random_flip : string, "h" | "hv" | "none" | "v"
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Default	: "hv"
    random_crop : string, "none" or "unique"
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions that
        are larger than those sizes are cropped.
        Default	: "unique"
    offsets : (double-1 <, double-2, ...>), optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default : (85, 111, 139)

    Returns
    -------
    :class:`Sequential`
        A model object using DenseNet_Cifar architecture.

    '''

    channel_in = conv_channel  # number of channel of transition conv layer

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(conv_channel, width=3, act='identity', includeBias=False, stride=1))

    for i in range(n_blocks):
        model.add(DenseNetBlock(n_cells=n_cells, kernel_size=3,
                                n_filter=growth_rate, stride=1))
        # transition block
        channel_in += (growth_rate * n_cells)
        model.add(BN(act='relu'))
        if i != (n_blocks - 1):
            model.add(Conv2d(channel_in, width=3, act='identity',
                             includeBias=False, stride=1))
            model.add(Pooling(width=2, height=2, pool='mean'))

    # Bottom Layers
    pooling_size = (width // (2 ** n_blocks // 2), height // (2 ** n_blocks // 2))
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model
