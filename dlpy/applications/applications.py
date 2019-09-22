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


from dlpy.sequential import Sequential
from dlpy.blocks import Bidirectional
from dlpy.layers import (InputLayer, Conv2d, Pooling, Dense, OutputLayer, Recurrent)
from dlpy.utils import DLPyError


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
        Default: GRU
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
        Default: GRU
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
        Default: GRU
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
        Default: GRU
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
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model
        will automatically detect the number of classes based on the
        training set.
        Default: 10
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 1
    width : int, optional
        Specifies the width of the input layer.
        Default: 28
    height : int, optional
        Specifies the height of the input layer.
        Default: 28
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
