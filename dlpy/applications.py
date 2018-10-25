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

from .sequential import Sequential
from .blocks import ResBlockBN, ResBlock_Caffe, DenseNetBlock, Bidirectional
from .caffe_models import (model_vgg16, model_vgg19, model_resnet50,
                           model_resnet101, model_resnet152)
from .keras_models import model_inceptionv3
from .layers import (InputLayer, Conv2d, Pooling, Dense, BN, OutputLayer, Detection, Concat, Reshape, Recurrent)
from .model import Model
from .utils import random_name, DLPyError


def TextClassification(conn, model_table='text_classifier', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a text classification model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 2:
        model = Sequential(conn=conn, model_table=model_table)
        b = Bidirectional(n=neurons, name='bi_'+rnn_type+'_layer_', n_blocks=n_blocks-1, rnn_type=rnn_type)
        model.add(b)
        model.add(Bidirectional(n=neurons, output_type='encoding', src_layers=b.get_last_layers(), rnn_type=rnn_type,
                                name='bi_'+rnn_type+'_lastlayer_',))
        model.add(OutputLayer())
    elif n_blocks == 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, output_type='encoding', rnn_type=rnn_type))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a text classification model should be at least 1.')

    return model


def TextGeneration(conn, model_table='text_generator', neurons=10, max_output_length=15, n_blocks=3, rnn_type='gru'):
    '''
    Generates a text generation model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    max_output_length : int, optional
        Specifies the maximum number of tokens to generate
        Default: 15
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 3:
        model = Sequential(conn=conn, model_table=model_table)
        b = Bidirectional(n=neurons, name='bi_'+rnn_type+'_layer_', n_blocks=n_blocks-2, rnn_type=rnn_type)
        model.add(b)
        b2 = Bidirectional(n=neurons, output_type='encoding', src_layers=b.get_last_layers(), rnn_type=rnn_type,
                           name='bi_'+rnn_type+'_lastlayer')
        model.add(b2)
        model.add(Recurrent(n=neurons, output_type='arbitrarylength', src_layers=b2.get_last_layers(),
                            rnn_type=rnn_type, max_output_length=max_output_length))
        model.add(OutputLayer())
    elif n_blocks >= 2:
        model = Sequential(conn=conn, model_table=model_table)
        b2 = Bidirectional(n=neurons, output_type='encoding', rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_')
        model.add(b2)
        model.add(Recurrent(n=neurons, output_type='arbitrarylength', src_layers=b2.get_last_layers(),
                            rnn_type=rnn_type, max_output_length=max_output_length))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a text generation model should be at least 2.')

    return model


def SequenceLabeling(conn, model_table='sequence_labeling_model', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a sequence labeling model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU
        
    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, n_blocks=n_blocks, rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_'))
        model.add(OutputLayer())
    else:
        raise DLPyError('The number of blocks for a sequence labeling model should be at least 1.')

    return model


def SpeechRecognition(conn, model_table='acoustic_model', neurons=10, n_blocks=3, rnn_type='gru'):
    '''
    Generates a speech recognition model.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    neurons : int, optional
        Specifies the number of neurons to be in each layer.
        Default: 10
    n_blocks : int, optional
        Specifies the number of bidirectional blocks to be added to the model.
        Default: 3
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: RNN
        Valid Values: RNN, LSTM, GRU

    Returns
    -------
    :class:`Sequential`

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if n_blocks >= 1:
        model = Sequential(conn=conn, model_table=model_table)
        model.add(Bidirectional(n=neurons, n_blocks=n_blocks, rnn_type=rnn_type, name='bi_'+rnn_type+'_layer_'))
        model.add(OutputLayer(error='CTC'))
    else:
        raise DLPyError('The number of blocks for an acoustic model should be at least 1.')

    return model


def LeNet5(conn, model_table='LENET5', n_classes=10, n_channels=1, width=28, height=28, scale=1.0 / 255,
           random_flip='none', random_crop='none', offsets=0):
    '''
    Generates a deep learning model with the LeNet5 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 1
    width : int, optional
        Specifies the width of the input layer.
        Default: 28
    height : int, optional
        Specifies the height of the input layer.
        Default: 28
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 10
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'v', 'hv', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data
        is used. Images are cropped to the values that are specified in the
        width and height parameters. Only the images with one or both
        dimensions that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: 0

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    model.add(Conv2d(n_filters=6, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=16, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=120))
    model.add(Dense(n=84))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG11(conn, model_table='VGG11', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the VGG11 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

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


def VGG13(conn, model_table='VGG13', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the VGG13 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG16(conn, model_table='VGG16', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the VGG16 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers)
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

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

        model.add(Dense(n=4096, dropout=0.5, name='fc6'))
        model.add(Dense(n=4096, dropout=0.5, name='fc7'))

        model.add(OutputLayer(n=n_classes, name='fc8'))

        return model

    else:
        # TODO: I need to re-factor loading / downloading pre-trained models.
        # something like pytorch style

        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg16.VGG16_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

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


def VGG19(conn, model_table='VGG19', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the VGG19 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers).
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

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
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg19.VGG19_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:

            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

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


def ResNet18_SAS(conn, model_table='RESNET18_SAS', batch_norm_first=True, n_classes=1000, n_channels=3, width=224,
                 height=224, scale=1, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet18 architecture.

    Compared to Caffe ResNet18, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                 batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet18_Caffe(conn, model_table='RESNET18_CAFFE', batch_norm_first=False, n_classes=1000, n_channels=3, width=224,
                   height=224, scale=1, random_flip='none', random_crop='none', offsets=None):
    '''
    Generates a deep learning model with the ResNet18 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first, conv_short_cut=conv_short_cut))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34_SAS(conn, model_table='RESNET34_SAS', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet34 architecture.

    Compared to Caffe ResNet34, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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


def ResNet34_Caffe(conn, model_table='RESNET34_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   batch_norm_first=False, random_flip='none', random_crop='none', offsets=None):
    '''
    Generates a deep learning model with the ResNet34 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
            model.add(ResBlock_Caffe(kernel_sizes=kernel_sizes, n_filters=n_filters, strides=strides,
                                     batch_norm_first=batch_norm_first, conv_short_cut=conv_short_cut))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_SAS(conn, model_table='RESNET50_SAS', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                 batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet50 architecture.

    Compared to Caffe ResNet50, the model prepends a batch normalization layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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

    model.add(BN(act='relu'))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_Caffe(conn, model_table='RESNET50_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                   batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                   pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet50 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: False
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        This option is required when pre_trained_weights=True.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers
        (i.e., the last layer for classification).
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_resnet50.ResNet50_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                                  width=width, height=height, random_crop=random_crop, offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)
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


def ResNet101_SAS(conn, model_table='RESNET101_SAS',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the ResNet101 architecture.

    Compared to Caffe ResNet101, the model prepends a batch normalization
    layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet101_Caffe(conn, model_table='RESNET101_CAFFE', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet101 architecture with convolution shortcut.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False.

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. specify the pre_trained_weights_file using the fully qualified server side path.')
        model_cas = model_resnet101.ResNet101_Model( s=conn, model_table=model_table, n_channels=n_channels,
                                                     width=width, height=height, random_crop=random_crop,
                                                     offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_trained_weights_file)
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


def ResNet152_SAS(conn, model_table='RESNET152_SAS',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  batch_norm_first=True, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the SAS ResNet152 architecture.

    Compared to Caffe ResNet152, the model prepends a batch normalization
    layer to the last global pooling layer.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))

    # Top layers
    model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
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
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                    height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet152_Caffe(conn, model_table='RESNET152_CAFFE',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                    batch_norm_first=False, random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the ResNet152 architecture with convolution shortcut

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
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
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the last layer for classification.
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is `False`
    :class:`Model`
        If `pre_trained_weights` is `True`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                             scale=scale, offsets=offsets, random_flip=random_flip,
                             random_crop=random_crop))
        # Top layers
        model.add(Conv2d(64, 7, act='identity', include_bias=False, stride=2))
        model.add(BN(act='relu'))
        model.add(Pooling(width=3, stride=2))
        # Residual block configuration.
        kernel_sizes_list = [(1, 3, 1)] * 4
        n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
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
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. specify the pre_trained_weights_file using the fully qualified server side path.')
        model_cas = model_resnet152.ResNet152_Model( s=conn, model_table=model_table, n_channels=n_channels,
                                                     width=width, height=height, random_crop=random_crop,
                                                     offsets=offsets)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(conn.CASTable(model_table), display_note=False)
            model.load_weights(path=pre_trained_weights_file)
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


def ResNet_Wide(conn, model_table='WIDE_RESNET', batch_norm_first=True, number_of_blocks=1, k=4, n_classes=None,
                n_channels=3, width=32, height=32, scale=1, random_flip='none', random_crop='none',
                offsets=(103.939, 116.779, 123.68)):
    '''
    Generate a deep learning model with Wide ResNet architecture.

    Wide ResNet is just a ResNet with more feature maps in each convolutional layers.
    The width of ResNet is controlled by widening factor k.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
    batch_norm_first : bool, optional
        Specifies whether to have batch normalization layer before the
        convolution layer in the residual block.  For a detailed discussion
        about this, please refer to this paper: He, Kaiming, et al. "Identity
        mappings in deep residual networks." European Conference on Computer
        Vision. Springer International Publishing, 2016.
        Default: True
    number_of_blocks : int
        Specifies the number of blocks in a residual group. For example,
        this value is [2, 2, 2, 2] for the ResNet18 architecture and [3, 4, 6, 3]
        for the ResNet34 architecture. In this case, the number of blocks
        are the same for each group as in the ResNet18 architecture.
        Default: 1
    k : int
        Specifies the widening factor.
        Default: 4
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 32
    height : int, optional
        Specifies the height of the input layer.
        Default: 32
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1605.07146.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    in_filters = 16

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale, offsets=offsets,
                         random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(in_filters, 3, act='identity', include_bias=False, stride=1))
    model.add(BN(act='relu'))

    # Residual block configuration.
    n_filters_list = [(16 * k, 16 * k), (32 * k, 32 * k), (64 * k, 64 * k)]
    kernel_sizes_list = [(3, 3)] * len(n_filters_list)
    rep_nums_list = [number_of_blocks, number_of_blocks, number_of_blocks]

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


def DenseNet(conn, model_table='DenseNet', n_classes=None, conv_channel=16, growth_rate=12, n_blocks=4,
             n_cells=4, n_channels=3, width=32, height=32, scale=1, random_flip='none', random_crop='none',
             offsets=(85, 111, 139)):
    '''
    Generates a deep learning model with the DenseNet architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: None
    conv_channel : int, optional
        Specifies the number of filters of the first convolution layer.
        Default: 16
    growth_rate : int, optional
        Specifies the growth rate of convolution layers.
        Default: 12
    n_blocks : int, optional
        Specifies the number of DenseNet blocks.
        Default: 4
    n_cells : int, optional
        Specifies the number of dense connection for each DenseNet block.
        Default: 4
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
        Default: 1.
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (85, 111, 139)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1608.06993.pdf

    '''

    channel_in = conv_channel  # number of channel of transition conv layer

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale,
                         offsets=offsets, random_flip=random_flip, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(conv_channel, width=3, act='identity', include_bias=False, stride=1))

    for i in range(n_blocks):
        model.add(DenseNetBlock(n_cells=n_cells, kernel_size=3, n_filter=growth_rate, stride=1))
        # transition block
        channel_in += (growth_rate * n_cells)
        model.add(BN(act='relu'))
        if i != (n_blocks - 1):
            model.add(Conv2d(channel_in, width=3, act='identity', include_bias=False, stride=1))
            model.add(Pooling(width=2, height=2, pool='mean'))

    # Bottom Layers
    pool_width = (width // (2 ** n_blocks // 2))
    if pool_width < 1:
        pool_width = 1
        print("WARNING: You seem to have a network that might be too deep for the input width.")

    pool_height = (height // (2 ** n_blocks // 2))
    if pool_height < 1:
        pool_height = 1
        print("WARNING: You seem to have a network that might be too deep for the input height.")

    model.add(Pooling(width=pool_width, height=pool_height, pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def DenseNet121(conn, model_table='DENSENET121', n_classes=1000, conv_channel=64, growth_rate=32,
                n_cells=[6, 12, 24, 16], n_channels=3, reduction=0.5, width=224, height=224, scale=1,
                random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68)):
    '''
    Generates a deep learning model with the DenseNet121 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: None
    conv_channel : int, optional
        Specifies the number of filters of the first convolution layer.
        Default: 16
    growth_rate : int, optional
        Specifies the growth rate of convolution layers.
        Default: 12
    n_cells : int, optional
        Specifies the number of dense connection for each DenseNet block.
        Default: 4
    reduction : double, optional
        Specifies the factor of transition blocks.
        Default: 0.5
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
        Default: 1.
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1608.06993.pdf

    '''
    n_blocks = len(n_cells)

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale,
                         random_flip=random_flip, offsets=offsets, random_crop=random_crop))
    # Top layers
    model.add(Conv2d(conv_channel, width=7, act='identity', include_bias=False, stride=2))
    model.add(BN(act='relu'))
    src_layer = Pooling(width=3, height=3, stride=2, padding=1, pool='max')
    model.add(src_layer)

    for i in range(n_blocks):
        for _ in range(n_cells[i]):

            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=growth_rate * 4, width=1, act='identity', stride=1, include_bias=False))

            model.add(BN(act='relu'))
            src_layer2 = Conv2d(n_filters=growth_rate, width=3, act='identity', stride=1, include_bias=False)

            model.add(src_layer2)
            src_layer = Concat(act='identity', src_layers=[src_layer, src_layer2])
            model.add(src_layer)

            conv_channel += growth_rate

        if i != (n_blocks - 1):
            # transition block
            conv_channel = int(conv_channel * reduction)

            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=conv_channel, width=1, act='identity', stride=1, include_bias=False))
            src_layer = Pooling(width=2, height=2, stride=2, pool='mean')

            model.add(src_layer)

    model.add(BN(act='identity'))
    # Bottom Layers
    pooling_size = (7, 7)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def Darknet_Reference(conn, model_table='Darknet_Reference', n_classes=1000, act='leaky',
                      n_channels=3, width=224, height=224, scale=1.0 / 255, random_flip='H', random_crop='UNIQUE'):

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
        Valid Values: 'none', 'unique'
        Default: 'unique'

    Returns
    -------
    :class:`Sequential`

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, scale=scale,
                         random_flip=random_flip, random_crop=random_crop))

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

    model.add(Pooling(width=7, height=7, pool='mean'))
    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def Darknet(conn, model_table='Darknet', n_classes=1000, act='leaky', n_channels=3, width=224, height=224,
            scale=1.0 / 255, random_flip='H', random_crop='UNIQUE'):
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
        Default: 1. / 255
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
        Valid Values: 'none', 'unique'
        Default: 'unique'

    Returns
    -------
    :class:`Sequential`

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, random_flip=random_flip,
                         random_crop=random_crop))
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

    model.add(Pooling(width=7, height=7, pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def YoloV2(conn, anchors, model_table='Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
           act='leaky', max_label_per_image=30, max_boxes=30, random_mutation='random', coord_type='YOLO',
           n_classes=20, predictions_per_grid=5, grid_number=13, **kwargs):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For
        example, if the value is 5, then the image will be divided into
        a 5 x 5 grid.
        Default: 13
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'random'
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    if len(anchors) != 2 * predictions_per_grid:
        raise DLPyError('The size of the anchor list in the detection layer for YOLOv2 should be equal to '
                        'twice the number of predictions_per_grid.')

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))

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
    model.add(BN(act=act))
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

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(detection_model_type='yolov2', class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, anchors=anchors, coord_type=coord_type,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes, **kwargs))

    return model


def YoloV2_MultiSize(conn, anchors, model_table='Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
                     act='leaky', coord_type='YOLO', max_label_per_image=30, max_boxes=30, random_mutation='random',
                     n_classes=20, predictions_per_grid=5, grid_number=13, **kwargs):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    The model is same as Yolov2 proposed in original paper. In addition to
    Yolov2, the model adds a passthrough layer that brings feature from an
    earlier layer to lower resolution layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'random'
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))

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
    pointLayer1 = BN(act=act, name='BN5_13')
    model.add(pointLayer1)
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
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act, name='BN6_19'))
    # conv20 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    pointLayer2 = BN(act=act, name='BN6_20')
    model.add(pointLayer2)

    # conv21 7 26 * 26 * 512 -> 26 * 26 * 64
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1, src_layers=[pointLayer1]))
    model.add(BN(act=act))
    # reshape 26 * 26 * 64 -> 13 * 13 * 256
    pointLayer3 = Reshape(act='identity', width=13, height=13, depth=256, name='reshape1')
    model.add(pointLayer3)

    # concat
    model.add(Concat(act='identity', src_layers=[pointLayer2, pointLayer3]))

    # conv22 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(detection_model_type='yolov2', class_number=n_classes, grid_number=grid_number,
                        coord_type=coord_type, predictions_per_grid=predictions_per_grid, anchors=anchors,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes, **kwargs))

    return model


def Tiny_YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
                act='leaky', coord_type='YOLO', max_label_per_image=30, max_boxes=30, random_mutation='random',
                n_classes=20, predictions_per_grid=5, grid_number=13, **kwargs):
    '''
    Generate a deep learning model with the Tiny Yolov2 architecture.

    Tiny Yolov2 is a very small model of Yolov2, so that it includes fewer
    numbers of convolutional layer and batch normalization layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    act : string, optional
        Specifies the activation function for the batch normalization layers.
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'random'
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))
    # conv1 416 448
    model.add(Conv2d(n_filters=16, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 208 224
    model.add(Conv2d(n_filters=32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 104 112
    model.add(Conv2d(n_filters=64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv4 52 56
    model.add(Conv2d(n_filters=128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv5 26 28
    model.add(Conv2d(n_filters=256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 13 14
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=1, pool='max'))
    # conv7 13
    model.add(Conv2d(n_filters=1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 13
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(detection_model_type='yolov2', class_number=n_classes, grid_number=grid_number,
                        coord_type=coord_type, predictions_per_grid=predictions_per_grid, anchors=anchors,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes, **kwargs))
    return model


def YoloV1(conn, model_table='Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255,
           n_classes=20, random_mutation='random', det_act='identity', act='leaky', dropout=0,
           predictions_per_grid=2, grid_number=7, **kwargs):
    '''
    Generates a deep learning model with the Yolo V1 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'random'
    det_act: string, optional
        Specifies the activation function for the detection layer.
        Default: 'identity'
    act: String, optional
        Specifies the activation function to be used in the batch normalization
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    predictions_per_grid: int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    grid_number: int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))
    # conv1 448
    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 224
    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    # conv4 112
    model.add(Conv2d(64, width=1, act=act, include_bias=False, stride=1))
    # conv5 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    # conv7 56
    model.add(Conv2d(128, width=1, act=act, include_bias=False, stride=1))
    # conv8 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv10 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv11 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv12 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv13 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv15 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv16 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv17 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv18 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))

    # conv19 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv20 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=2))
    # conv21 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv22 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv23 7
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))
    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act=det_act))

    model.add(Detection(detection_model_type='yolov1', class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, **kwargs))

    return model


def Tiny_YoloV1(conn, model_table='Tiny-Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255, act='leaky',
                coord_type='YOLO', random_mutation='random', dropout=0, n_classes=10, predictions_per_grid=2,
                grid_number=7, **kwargs):
    '''
    Generates a deep learning model with the Tiny Yolov1 architecture.

    Tiny Yolov1 is a very small model of Yolov1, so that it includes
    fewer numbers of convolutional layer.

    Parameters
    -----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the training set.
        Default: 10
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For
        example, if the value is 5, then the image will be divided into
        a 5 x 5 grid.
        Default: 7
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    act : string, optional
        Specifies the activation function for the convolutional layers.
        Default: 'leaky'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'random'
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    dropout : double, optional
        Specifies the dropout rate.
        Default: 0

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height, random_mutation=random_mutation,
                         scale=scale))

    model.add(Conv2d(16, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))

    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(detection_model_type='yolov1', class_number=n_classes, grid_number=grid_number,
                        coord_type=coord_type, predictions_per_grid=predictions_per_grid, **kwargs))

    return model


def InceptionV3(conn, model_table='InceptionV3',
                n_classes=1000, n_channels=3, width=299, height=299, scale=1,
                random_flip='none', random_crop='none', offsets=(103.939, 116.779, 123.68),
                pre_trained_weights=False, pre_trained_weights_file=None, include_top=False):
    '''
    Generates a deep learning model with the Inceptionv3 architecture with batch normalization layers.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model in.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 299
    height : int, optional
        Specifies the height of the input layer.
        Default: 299
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
        Default: 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
        Note: Required when pre_train_weight=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the FC layers
        Default: False

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    References
    ----------
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        model.add(InputLayer(n_channels=n_channels, width=width,
                             height=height, scale=scale, offsets=offsets,
                             random_flip=random_flip, random_crop=random_crop))

        # 299 x 299 x 3
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 149 x 149 x 32
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 147 x 147 x 32
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        # 147 x 147 x 64
        model.add(Pooling(width=3, height=3, stride=2, pool='max', padding=0))

        # 73 x 73 x 64
        model.add(Conv2d(n_filters=80, width=1, height=1, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 73 x 73 x 80
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 71 x 71 x 192
        pool2 = Pooling(width=3, height=3, stride=2, pool='max', padding=0)
        model.add(pool2)


        # mixed 0: output 35 x 35 x 256

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[pool2]))
        model.add(Conv2d(n_filters=32, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed0 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 1: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed1 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 2: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed2 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 3: output 17 x 17 x 768

        # branch3x3
        model.add(Conv2d(n_filters=384, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0,
                         src_layers=[concat]))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed3 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch3x3dbl, branch_pool])
        model.add(concat)


        # mixed 4: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed4 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 5, 6: output 17 x 17 x 768
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch7x7
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            branch7x7 = BN(act='relu')
            model.add(branch7x7)

            # branch7x7dbl
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            branch7x7dbl = BN(act='relu')
            model.add(branch7x7dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                        branch_pool])
            model.add(concat)


        # mixed 7: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed7 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 8: output 8 x 8 x 1280

        # branch3x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=320, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch7x7x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch7x7x3 = BN(act='relu')
        model.add(branch7x7x3)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed8 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch7x7x3, branch_pool])
        model.add(concat)


        # mixed 9, 10:  output 8 x 8 x 2048
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=320, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch3x3
            model.add(Conv2d(n_filters=384, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch3x3 = BN(act='relu')
            model.add(branch3x3)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_1 = BN(act='relu')
            model.add(branch3x3_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_2 = BN(act='relu')
            model.add(branch3x3_2)

            branch3x3 = Concat(act='identity',
                               src_layers=[branch3x3_1, branch3x3_2])
            model.add(branch3x3)

            # branch3x3dbl
            model.add(Conv2d(n_filters=448, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=384, width=3, height=3, stride=1,
                             act='identity', include_bias=False))
            branch3x3dbl = BN(act='relu')
            model.add(branch3x3dbl)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_1 = BN(act='relu')
            model.add(branch3x3dbl_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_2 = BN(act='relu')
            model.add(branch3x3dbl_2)

            branch3x3dbl = Concat(act='identity',
                                  src_layers=[branch3x3dbl_1, branch3x3dbl_2])
            model.add(branch3x3dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch3x3,
                                        branch3x3dbl, branch_pool])
            model.add(concat)


        # calculate dimensions for global average pooling
        w = max((width - 75) // 32 + 1, 1)
        h = max((height - 75) // 32 + 1, 1)

        # global average pooling
        model.add(Pooling(width=w, height=h, stride=1, pool='average',
                          padding=0, src_layers=[concat]))

        # output layer
        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the '
                             'pre-trained weights:\n'
                             '1. go to the website '
                             'https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. upload the *.h5 file to '
                             'a server side directory which the CAS '
                             'session has access to.\n'
                             '3. specify the pre_train_weight_file using '
                             'the fully qualified server side path.')
        print('NOTE: Scale is set to 1/127.5, and offsets 1 to '
              'match Keras preprocessing.')
        model_cas = model_inceptionv3.InceptionV3_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=[1, 1, 1])

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, '
                              'n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<218'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True,
                                         **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table,
                             name='predictions')
            model._retrieve_('deeplearn.addlayer', model=model_table,
                             name='predictions',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['avg_pool'])
            model = Model.from_table(conn.CASTable(model_table))

            return model