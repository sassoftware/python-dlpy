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

import warnings
import six

from .sequential import Sequential
from .blocks import ResBlockBN, ResBlock_Caffe, DenseNetBlock, Bidirectional
from .caffe_models import (model_vgg16, model_vgg19, model_resnet50,
                           model_resnet101, model_resnet152)
from .keras_models import model_inceptionv3
from .layers import (Input, InputLayer, Conv2d, Pooling, Dense, BN, OutputLayer, Detection, Concat, Reshape, Recurrent,
                     RegionProposal, ROIPooling, FastRCNN, GlobalAveragePooling2D, GroupConv2d, ChannelShuffle, Res,
                     Segmentation, Conv2DTranspose)
from .model import Model
from .utils import random_name, DLPyError

# input layer option will be found in model function's local parameters
input_layer_options = ['n_channels', 'width', 'height', 'nominals', 'std', 'scale', 'offsets',
                       'dropout', 'random_flip', 'random_crop', 'random_mutation', 'norm_stds']
# RPN layer option will be found in model function's local parameters
rpn_layer_options = ['anchor_ratio', 'anchor_scale', 'anchor_num_to_sample', 'base_anchor_size',
                     'coord_type', 'do_RPN_only', 'max_label_per_image', 'proposed_roi_num_score',
                     'proposed_roi_num_train', 'roi_train_sample_num']
# Fast RCNN option will be found in model function's local parameters
fast_rcnn_options = ['detection_threshold', 'max_label_per_image', 'max_object_num', 'nms_iou_threshold']


def _get_layer_options(layer_options, local_options):
    """
    Get parameters belonging to a certain type of layer.

    Parameters
    ----------
    layer_options : list of String
        Specifies parameters of the layer.
    local_options : list of dictionary
        Specifies local parameters in a model function.

    """
    layer_options_dict = {}
    for key, value in six.iteritems(local_options):
        if key in layer_options:
            layer_options_dict[key] = value
    return layer_options_dict


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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'none'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: None

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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50_Caffe(conn, model_table='RESNET50_CAFFE', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
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
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
        model.add(GlobalAveragePooling2D())

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
        model.add(GlobalAveragePooling2D())

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
        model.add(GlobalAveragePooling2D())

        model.add(OutputLayer(act='softmax', n=n_classes))

        return model
    else:
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the pre-trained weights:\n'
                             '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. Upload the *.h5 file to '
                             'a server side directory which the CAS session has access to.\n'
                             '3. Specify the pre_trained_weights_file using the fully qualified server side path.')
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
        Default: None
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def MobileNetV1(conn, model_table='MobileNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                random_flip='none', random_crop='none', random_mutation='none',
                norm_stds=(255*0.229, 255*0.224, 255*0.225), offsets=(255*0.485, 255*0.456, 255*0.406),
                alpha=1, depth_multiplier=1):
    '''
    Generates a deep learning model with the MobileNetV1 architecture.
    The implementation is revised based on
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
        Default: (255*0.229, 255*0.224, 255*0.225)
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (255*0.485, 255*0.456, 255*0.406)
    alpha : int, optional
        Specifies the width multiplier in the MobileNet paper
        Default: 1
    depth_multiplier : int, optional
        Specifies the number of depthwise convolution output channels for each input channel.
        Default: 1

    Returns
    -------
    :class:`Model`

    References
    ----------
    https://arxiv.org/pdf/1605.07146.pdf

    '''
    def _conv_block(inputs, filters, alpha, kernel=3, stride=1):
        """
        Adds an initial convolution layer (with batch normalization

        inputs:
            Input tensor
        filters:
            the dimensionality of the output space
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel:
            specifying the width and height of the 2D convolution window.
        strides:
            the strides of the convolution

        """
        filters = int(filters * alpha)
        x = Conv2d(filters, kernel, act = 'identity', include_bias = False, stride = stride, name = 'conv1')(inputs)
        x = BN(name = 'conv1_bn', act='relu')(x)
        return x, filters

    def _depthwise_conv_block(inputs, n_groups, pointwise_conv_filters, alpha,
                              depth_multiplier = 1, stride = 1, block_id = 1):
        """Adds a depthwise convolution block.

        inputs:
            Input tensor
        n_groups : int
            number of groups
        pointwise_conv_filters:
            the dimensionality of the output space
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier:
            The number of depthwise convolution output channels
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution
        block_id: Integer, a unique identification designating
            the block number.

        """
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        x = GroupConv2d(n_groups*depth_multiplier, n_groups, 3, stride = stride, act = 'identity',
                        include_bias = False, name = 'conv_dw_%d' % block_id)(inputs)
        x = BN(name = 'conv_dw_%d_bn' % block_id, act = 'relu')(x)

        x = Conv2d(pointwise_conv_filters, 1, act='identity', include_bias=False, stride=1,
                   name='conv_pw_%d' % block_id)(x)
        x = BN(name='conv_pw_%d_bn' % block_id, act='relu')(x)
        return x, pointwise_conv_filters

    parameters = locals()
    input_parameters = _get_layer_options(input_layer_options, parameters)
    inp = Input(**input_parameters, name = 'data')
    # the model down-sampled for 5 times by performing stride=2 convolution on
    # conv_dw_1, conv_dw_2, conv_dw_4, conv_dw_6, conv_dw_12
    # for each block, we use depthwise convolution with kernel=3 and point-wise convolution to save computation
    x, depth = _conv_block(inp, 32, alpha, stride=2)
    x, depth = _depthwise_conv_block(x, depth, 64, alpha, depth_multiplier, block_id=1)

    x, depth = _depthwise_conv_block(x, depth, 128, alpha, depth_multiplier,
                                     stride=2, block_id=2)
    x, depth = _depthwise_conv_block(x, depth, 128, alpha, depth_multiplier, block_id=3)

    x, depth = _depthwise_conv_block(x, depth, 256, alpha, depth_multiplier,
                                     stride=2, block_id=4)
    x, depth = _depthwise_conv_block(x, depth, 256, alpha, depth_multiplier, block_id=5)

    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier,
                                     stride=2, block_id=6)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id=7)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id=8)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id=9)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id=10)
    x, depth = _depthwise_conv_block(x, depth, 512, alpha, depth_multiplier, block_id=11)

    x, depth = _depthwise_conv_block(x, depth, 1024, alpha, depth_multiplier,
                                     stride=2, block_id=12)
    x, depth = _depthwise_conv_block(x, depth, 1024, alpha, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D(name="Global_avg_pool")(x)
    x = OutputLayer(n=n_classes)(x)

    model = Model(conn, inp, x, model_table)
    model.compile()

    return model


def MobileNetV2(conn, model_table='MobileNetV2', n_classes=1000, n_channels=3, width=224, height=224,
                norm_stds=(255*0.229, 255*0.224, 255*0.225), offsets=(255*0.485, 255*0.456, 255*0.406),
                random_flip='none', random_crop='none', random_mutation='none', alpha=1):
    '''
    Generates a deep learning model with the MobileNetV2 architecture.
    The implementation is revised based on
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
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
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
        Default: (255 * 0.229, 255 * 0.224, 255 * 0.225)
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (255*0.485, 255*0.456, 255*0.406)
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    alpha : int, optional
        Specifies the width multiplier in the MobileNet paper
        Default: 1

    alpha : int, optional

    Returns
    -------
    :class:`Model`

    References
    ----------
    https://arxiv.org/abs/1801.04381

    '''
    def _make_divisible(v, divisor, min_value=None):
        # make number of channel divisible
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _inverted_res_block(inputs, in_channels, expansion, stride, alpha, filters, block_id):
        """
        Inverted Residual Block

        Parameters
        ----------
        inputs:
            Input tensor
        in_channels:
            Specifies the number of input tensor's channel
        expansion:
            expansion factor always applied to the input size.
        stride:
            the strides of the convolution
        alpha:
            width multiplier.
        filters:
            the dimensionality of the output space.
        block_id:
            block id used for naming layers

        """
        pointwise_conv_filters = int(filters * alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        x = inputs
        prefix = 'block_{}_'.format(block_id)
        n_groups = in_channels

        if block_id:
            # Expand
            n_groups = expansion * in_channels
            x = Conv2d(expansion * in_channels, 1, include_bias=False, act='identity',
                       name = prefix + 'expand')(x)
            x = BN(name = prefix + 'expand_BN', act='identity')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        x = GroupConv2d(n_groups, n_groups, 3, stride=stride, act='identity',
                        include_bias=False, name=prefix + 'depthwise')(x)
        x = BN(name = prefix + 'depthwise_BN', act='relu')(x)

        # Project
        x = Conv2d(pointwise_filters, 1, include_bias=False, act='identity', name=prefix + 'project')(x)
        x = BN(name=prefix + 'project_BN', act='identity')(x)  # identity activation on narrow tensor

        if in_channels == pointwise_filters and stride == 1:
            return Res(name=prefix + 'add')([inputs, x]), pointwise_filters
        return x, pointwise_filters

    parameters = locals()
    input_parameters = _get_layer_options(input_layer_options, parameters)
    inp = Input(**input_parameters, name='data')
    # compared with mobilenetv1, v2 introduces inverted residual structure.
    # and Non-linearities in narrow layers are removed.
    # inverted residual block does three convolutins: first is 1*1 convolution, second is depthwise convolution,
    # third is 1*1 convolution but without any non-linearity
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2d(first_block_filters, 3, stride=2, include_bias=False, name='Conv1', act='identity')(inp)
    x = BN(name='bn_Conv1', act='relu')(x)

    x, n_channels = _inverted_res_block(x, first_block_filters, filters=16, alpha=alpha, stride=1,
                                        expansion=1, block_id=0)

    x, n_channels = _inverted_res_block(x, n_channels, filters=24, alpha=alpha, stride=2,
                                        expansion=6, block_id=1)
    x, n_channels = _inverted_res_block(x, n_channels, filters=24, alpha=alpha, stride=1,
                                        expansion=6, block_id=2)

    x, n_channels = _inverted_res_block(x, n_channels, filters=32, alpha=alpha, stride=2,
                                        expansion=6, block_id=3)
    x, n_channels = _inverted_res_block(x, n_channels, filters=32, alpha=alpha, stride=1,
                                        expansion=6, block_id=4)
    x, n_channels = _inverted_res_block(x, n_channels, filters=32, alpha=alpha, stride=1,
                                        expansion=6, block_id=5)

    x, n_channels = _inverted_res_block(x, n_channels, filters=64, alpha=alpha, stride=2,
                                        expansion=6, block_id=6)
    x, n_channels = _inverted_res_block(x, n_channels, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=7)
    x, n_channels = _inverted_res_block(x, n_channels, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=8)
    x, n_channels = _inverted_res_block(x, n_channels, filters=64, alpha=alpha, stride=1,
                                        expansion=6, block_id=9)

    x, n_channels = _inverted_res_block(x, n_channels, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=10)
    x, n_channels = _inverted_res_block(x, n_channels, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=11)
    x, n_channels = _inverted_res_block(x, n_channels, filters=96, alpha=alpha, stride=1,
                                        expansion=6, block_id=12)

    x, n_channels = _inverted_res_block(x, n_channels, filters=160, alpha=alpha, stride=2,
                                        expansion=6, block_id=13)
    x, n_channels = _inverted_res_block(x, n_channels, filters=160, alpha=alpha, stride=1,
                                        expansion=6, block_id=14)
    x, n_channels = _inverted_res_block(x, n_channels, filters=160, alpha=alpha, stride=1,
                                        expansion=6, block_id=15)

    x, n_channels = _inverted_res_block(x, n_channels, filters=320, alpha=alpha, stride=1,
                                        expansion=6, block_id=16)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = Conv2d(last_block_filters, 1, include_bias=False, name='Conv_1', act='identity')(x)
    x = BN(name='Conv_1_bn', act='relu')(x)

    x = GlobalAveragePooling2D(name="Global_avg_pool")(x)
    x = OutputLayer(n=n_classes)(x)

    model = Model(conn, inp, x, model_table)
    model.compile()

    return model


def ShuffleNetV1(conn, model_table='ShuffleNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                 norm_stds=(255*0.229, 255*0.224, 255*0.225), offsets=(255*0.485, 255*0.456, 255*0.406),
                 random_flip='none', random_crop='none', random_mutation='none', scale_factor=1.0,
                 num_shuffle_units=[3, 7, 3], bottleneck_ratio=0.25, groups=3, block_act='identity'):
    '''
    Generates a deep learning model with the ShuffleNetV1 architecture.
    The implementation is revised based on https://github.com/scheckmedia/keras-shufflenet/blob/master/shufflenet.py

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
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
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
        Default: (255 * 0.229, 255 * 0.224, 255 * 0.225)
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (255*0.485, 255*0.456, 255*0.406)
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
        Default: 'none'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    scale_factor : double

    num_shuffle_units: iter-of-int, optional
        number of stages (list length) and the number of shufflenet units in a
        stage beginning with stage 2 because stage 1 is fixed
        e.g. idx 0 contains 3 + 1 (first shuffle unit in each stage differs) shufflenet units for stage 2
        idx 1 contains 7 + 1 Shufflenet Units for stage 3 and
        idx 2 contains 3 + 1 Shufflenet Units
        Default: [3, 7, 3]
    bottleneck_ratio : double
        bottleneck ratio implies the ratio of bottleneck channels to output channels.
        For example, bottleneck ratio = 1 : 4 means the output feature map is 4 times
        the width of the bottleneck feature map.
    groups: int
        Specifies the number of groups per channel
        Default : 3
    block_act : str
        Specifies the activation function after depth-wise convolution and batch normalization layer
        Default : 'identity'

    Returns
    -------
    :class:`Model`

    References
    ----------
    https://arxiv.org/pdf/1707.01083

    '''

    def _block(x, channel_map, bottleneck_ratio, repeat = 1, groups = 1, stage = 1):
        """
        creates a bottleneck block

        Parameters
        ----------
        x:
            Input tensor
        channel_map:
            list containing the number of output channels for a stage
        repeat:
            number of repetitions for a shuffle unit with stride 1
        groups:
            number of groups per channel
        bottleneck_ratio:
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
        stage:
            stage number

        Returns
        -------
        """
        x = _shuffle_unit(x, in_channels = channel_map[stage - 2],
                          out_channels = channel_map[stage - 1], strides = 2,
                          groups = groups, bottleneck_ratio = bottleneck_ratio,
                          stage = stage, block = 1)

        for i in range(1, repeat + 1):
            x = _shuffle_unit(x, in_channels = channel_map[stage - 1],
                              out_channels = channel_map[stage - 1], strides = 1,
                              groups = groups, bottleneck_ratio = bottleneck_ratio,
                              stage = stage, block = (i + 1))

        return x

    def _shuffle_unit(inputs, in_channels, out_channels, groups, bottleneck_ratio, strides = 2, stage = 1, block = 1):
        """
        create a shuffle unit

        Parameters
        ----------
        inputs:
            Input tensor of with `channels_last` data format
        in_channels:
            number of input channels
        out_channels:
            number of output channels
        strides:
            An integer or tuple/list of 2 integers,
        groups:
            number of groups per channel
        bottleneck_ratio: float
            bottleneck ratio implies the ratio of bottleneck channels to output channels.
        stage:
            stage number
        block:
            block number

        """
        prefix = 'stage%d/block%d' % (stage, block)

        # if strides >= 2:
        # out_channels -= in_channels

        # default: 1/4 of the output channel of a ShuffleNet Unit
        bottleneck_channels = int(out_channels * bottleneck_ratio)
        groups = (1 if stage == 2 and block == 1 else groups)

        # x = _group_conv(inputs, in_channels, out_channels = bottleneck_channels,
        #                 groups = (1 if stage == 2 and block == 1 else groups),
        #                 name = '%s/1x1_gconv_1' % prefix)

        x = GroupConv2d(bottleneck_channels, n_groups = (1 if stage == 2 and block == 1 else groups), act = 'identity',
                        width = 1, height = 1, stride = 1, include_bias = False,
                        name = '%s/1x1_gconv_1' % prefix)(inputs)

        x = BN(act = 'relu', name = '%s/bn_gconv_1' % prefix)(x)

        x = ChannelShuffle(n_groups = groups, name = '%s/channel_shuffle' % prefix)(x)
        # depthwise convolutioin
        x = GroupConv2d(x.shape[-1], n_groups = x.shape[-1], width = 3, height = 3, include_bias = False,
                        stride = strides, act = 'identity',
                        name = '%s/1x1_dwconv_1' % prefix)(x)
        x = BN(act = block_act, name = '%s/bn_dwconv_1' % prefix)(x)

        out_channels = out_channels if strides == 1 else out_channels - in_channels
        x = GroupConv2d(out_channels, n_groups = groups, width = 1, height = 1, stride=1, act = 'identity',
                        include_bias = False, name = '%s/1x1_gconv_2' % prefix)(x)

        x = BN(act = block_act, name = '%s/bn_gconv_2' % prefix)(x)

        if strides < 2:
            ret = Res(act = 'relu', name = '%s/add' % prefix)([x, inputs])
        else:
            avg = Pooling(width = 3, height = 3, stride = 2, pool = 'mean', name = '%s/avg_pool' % prefix)(inputs)
            ret = Concat(act = 'relu', name = '%s/concat' % prefix)([x, avg])

        return ret

    out_dim_stage_two = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}
    try:
        import numpy as np
    except:
        raise DLPyError('Please install numpy to use this architecture.')

    exp = np.insert(np.arange(0, len(num_shuffle_units), dtype = np.float32), 0, 0)
    out_channels_in_stage = 2 ** exp
    out_channels_in_stage *= out_dim_stage_two[groups]  # calculate output channels for each stage
    out_channels_in_stage[0] = 24  # first stage has always 24 output channels
    out_channels_in_stage *= scale_factor
    out_channels_in_stage = out_channels_in_stage.astype(int)

    parameters = locals()
    input_parameters = _get_layer_options(input_layer_options, parameters)
    inp = Input(**input_parameters, name = 'data')

    # create shufflenet architecture
    x = Conv2d(out_channels_in_stage[0], 3, include_bias=False, stride=2, act="identity", name="conv1")(inp)
    x = BN(act='relu', name = 'bn1')(x)
    x = Pooling(width=3, height=3, stride=2, name="maxpool1")(x)

    # create stages containing shufflenet units beginning at stage 2
    for stage in range(0, len(num_shuffle_units)):
        repeat = num_shuffle_units[stage]
        x = _block(x, out_channels_in_stage, repeat=repeat,
                   bottleneck_ratio=bottleneck_ratio,
                   groups=groups, stage=stage + 2)

    x = GlobalAveragePooling2D(name="Global_avg_pool")(x)
    x = OutputLayer(n=n_classes)(x)

    model = Model(conn, inputs=inp, outputs=x, model_table=model_table)
    model.compile()

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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

    model.add(GlobalAveragePooling2D())

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
        Default: 1000
    conv_channel : int, optional
        Specifies the number of filters of the first convolution layer.
        Default: 64
    growth_rate : int, optional
        Specifies the growth rate of convolution layers.
        Default: 32
    n_cells : int array length=4, optional
        Specifies the number of dense connection for each DenseNet block.
        Default: [6, 12, 24, 16]
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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    model.add(GlobalAveragePooling2D())

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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

    model.add(GlobalAveragePooling2D())
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

    model.add(GlobalAveragePooling2D())

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
           random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
           num_to_force_coord=None):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. 
        Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

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

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))

    return model


def YoloV2_MultiSize(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
                     random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                     coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                     n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                     coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                     detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                     num_to_force_coord=None):
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
    model_table : string, optional
        Specifies the name of CAS table to store the model.
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

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
    pointLayer3 = Reshape(act='identity', width=grid_number, height=grid_number, depth=256, name='reshape1')
    model.add(pointLayer3)

    # concat
    model.add(Concat(act='identity', src_layers=[pointLayer2, pointLayer3]))

    # conv22 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))

    return model


def Tiny_YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
                random_mutation='NONE', act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                num_to_force_coord=None):
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
    model_table : string, optional
        Specifies the name of CAS table to store the model.
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. 
        Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center

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

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))
    return model


def YoloV1(conn, model_table='Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255,
           random_mutation='NONE', act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False):
    '''
    Generates a deep learning model with the Yolo V1 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. 
        Default: False

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
    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov1',
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes))

    return model


def Tiny_YoloV1(conn, model_table='Tiny-Yolov1', n_channels=3, width=448, height=448, scale=1.0 / 255,
                random_mutation='NONE', act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False):
    '''
    Generates a deep learning model with the Tiny Yolov1 architecture.

    Tiny Yolov1 is a very small model of Yolov1, so that it includes
    fewer numbers of convolutional layer.

        Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. 
        Default: False

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

    model.add(Detection(act=act_detection, detection_model_type='yolov1',
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes))

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
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
                             '1. Go to the website '
                             'https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. Upload the *.h5 file to '
                             'a server side directory which the CAS '
                             'session has access to.\n'
                             '3. Specify the pre_train_weight_file using '
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


def UNet(conn, model_table='UNet', n_classes = 2, n_channels=1, width=256, height=256, scale=1.0/255,
         norm_stds=None, offsets=None, random_mutation=None, init=None, bn_after_convolutions=False):
    '''
    Generates a deep learning model with the U-Net architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 2
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 256
    height : int, optional
        Specifies the height of the input layer.
        Default: 256
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0/255
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: NONE
    init : str
        Specifies the initialization scheme for convolution layers.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: None
    bn_after_convolutions : Boolean
        If set to True, a batch normalization layer is added after each convolution layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1505.04597

    '''
    parameters = locals()
    input_parameters = _get_layer_options(input_layer_options, parameters)
    inp = Input(**input_parameters, name = 'data')
    act_conv = 'relu'
    bias_conv = True
    if bn_after_convolutions:
        act_conv = 'identity'
        bias_conv = False
    # The model follows UNet paper architecture. The network down-samples by performing max pooling with stride=2
    conv1 = Conv2d(64, 3, act = act_conv, init = init, include_bias = bias_conv)(inp)
    conv1 = BN(act = 'relu')(conv1) if bn_after_convolutions else conv1
    conv1 = Conv2d(64, 3, act = act_conv, init = init, include_bias = bias_conv)(conv1)
    conv1 = BN(act = 'relu')(conv1) if bn_after_convolutions else conv1
    pool1 = Pooling(2)(conv1)

    conv2 = Conv2d(128, 3, act = act_conv, init = init, include_bias = bias_conv)(pool1)
    conv2 = BN(act = 'relu')(conv2) if bn_after_convolutions else conv2
    conv2 = Conv2d(128, 3, act = act_conv, init = init, include_bias = bias_conv)(conv2)
    conv2 = BN(act = 'relu')(conv2) if bn_after_convolutions else conv2
    pool2 = Pooling(2)(conv2)

    conv3 = Conv2d(256, 3, act = act_conv, init = init, include_bias = bias_conv)(pool2)
    conv3 = BN(act = 'relu')(conv3) if bn_after_convolutions else conv3
    conv3 = Conv2d(256, 3, act = act_conv, init = init, include_bias = bias_conv)(conv3)
    conv3 = BN(act = 'relu')(conv3) if bn_after_convolutions else conv3
    pool3 = Pooling(2)(conv3)

    conv4 = Conv2d(512, 3, act = act_conv, init = init, include_bias = bias_conv)(pool3)
    conv4 = BN(act = 'relu')(conv4) if bn_after_convolutions else conv4
    conv4 = Conv2d(512, 3, act = act_conv, init = init, include_bias = bias_conv)(conv4)
    conv4 = BN(act = 'relu')(conv4) if bn_after_convolutions else conv4
    pool4 = Pooling(2)(conv4)

    conv5 = Conv2d(1024, 3, act = act_conv, init = init, include_bias = bias_conv)(pool4)
    conv5 = BN(act = 'relu')(conv5) if bn_after_convolutions else conv5
    conv5 = Conv2d(1024, 3, act = act_conv, init = init, include_bias = bias_conv)(conv5)
    conv5 = BN(act = 'relu')(conv5) if bn_after_convolutions else conv5
    # the minimum is 1/2^4 of the original image size
    # Our implementation applies Transpose convolution to upsample feature maps.
    tconv6 = Conv2DTranspose(512, 3, stride = 2, act = 'relu', padding = 1, output_size = conv4.shape,
                             init = init)(conv5)  # 64
    # concatenation layers to combine encoder and decoder features
    merge6 = Concat()([conv4, tconv6])
    conv6 = Conv2d(512, 3, act = act_conv, init = init, include_bias = bias_conv)(merge6)
    conv6 = BN(act = 'relu')(conv6) if bn_after_convolutions else conv6
    conv6 = Conv2d(512, 3, act = act_conv, init = init, include_bias = bias_conv)(conv6)
    conv6 = BN(act = 'relu')(conv6) if bn_after_convolutions else conv6

    tconv7 = Conv2DTranspose(256, 3, stride = 2, act = 'relu', padding = 1, output_size = conv3.shape,
                             init = init)(conv6)  # 128
    merge7 = Concat()([conv3, tconv7])
    conv7 = Conv2d(256, 3, act = act_conv, init = init, include_bias = bias_conv)(merge7)
    conv7 = BN(act = 'relu')(conv7) if bn_after_convolutions else conv7
    conv7 = Conv2d(256, 3, act = act_conv, init = init, include_bias = bias_conv)(conv7)
    conv7 = BN(act = 'relu')(conv7) if bn_after_convolutions else conv7

    tconv8 = Conv2DTranspose(128, stride = 2, act = 'relu', padding = 1, output_size = conv2.shape,
                             init = init)(conv7)  # 256
    merge8 = Concat()([conv2, tconv8])
    conv8 = Conv2d(128, 3, act = act_conv, init = init, include_bias = bias_conv)(merge8)
    conv8 = BN(act = 'relu')(conv8) if bn_after_convolutions else conv8
    conv8 = Conv2d(128, 3, act = act_conv, init = init, include_bias = bias_conv)(conv8)
    conv8 = BN(act = 'relu')(conv8) if bn_after_convolutions else conv8

    tconv9 = Conv2DTranspose(64, stride = 2, act = 'relu', padding = 1, output_size = conv1.shape,
                             init = init)(conv8)  # 512
    merge9 = Concat()([conv1, tconv9])
    conv9 = Conv2d(64, 3, act = act_conv, init = init, include_bias = bias_conv)(merge9)
    conv9 = BN(act = 'relu')(conv9) if bn_after_convolutions else conv9
    conv9 = Conv2d(64, 3, act = act_conv, init = init, include_bias = bias_conv)(conv9)
    conv9 = BN(act = 'relu')(conv9) if bn_after_convolutions else conv9

    conv9 = Conv2d(n_classes, 3, act = 'relu', init = init)(conv9)

    seg1 = Segmentation(name = 'Segmentation_1')(conv9)
    model = Model(conn, inputs = inp, outputs = seg1, model_table = model_table)
    model.compile()
    return model


def Faster_RCNN(conn, model_table='Faster_RCNN', n_channels=3, width=1000, height=496, scale=1,
                norm_stds=None, offsets=(102.9801, 115.9465, 122.7717), random_mutation = 'none',
                n_classes=20, anchor_num_to_sample=256, anchor_ratio=[0.5, 1, 2], anchor_scale=[8, 16, 32],
                base_anchor_size=16, coord_type='coco', max_label_per_image=200, proposed_roi_num_train=2000,
                proposed_roi_num_score=300, roi_train_sample_num=128, roi_pooling_height=7, roi_pooling_width=7,
                nms_iou_threshold=0.3, detection_threshold=0.5, max_object_num=50):
    '''
    Generates a deep learning model with the faster RCNN architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 1000
    height : int, optional
        Specifies the height of the input layer.
        Default: 496
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
        Default: 'NONE'
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    anchor_num_to_sample : int, optional
        Specifies the number of anchors to sample for training the region proposal network
        Default: 256
    anchor_ratio : iter-of-float
        Specifies the anchor height and width ratios (h/w) used.
    anchor_scale : iter-of-float
        Specifies the anchor scales used based on base_anchor_size
    base_anchor_size : int, optional
        Specifies the basic anchor size in width and height (in pixels) in the original input image dimension
        Default: 16
    coord_type : int, optional
        Specifies the coordinates format type in the input label and detection result.
        Valid Values: RECT, COCO, YOLO
        Default: COCO
    proposed_roi_num_score: int, optional
        Specifies the number of ROI (Region of Interest) to propose in the scoring phase
        Default: 300
    proposed_roi_num_train: int, optional
        Specifies the number of ROI (Region of Interest) to propose used for RPN training, and also the pool to
        sample from for FastRCNN Training in the training phase
        Default: 2000
    roi_train_sample_num: int, optional
        Specifies the number of ROIs(Regions of Interests) to sample after NMS(Non-maximum Suppression)
        is performed in the training phase.
        Default: 128
    roi_pooling_height : int, optional
        Specifies the output height of the region pooling layer.
        Default: 7
    roi_pooling_width : int, optional
        Specifies the output width of the region pooling layer.
        Default: 7
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 200
    nms_iou_threshold: float, optional
        Specifies the IOU threshold of maximum suppression in object detection
        Default: 0.3
    detection_threshold : float, optional
        Specifies the threshold for object detection.
        Default: 0.5
    max_object_num: int, optional
        Specifies the maximum number of object to detect
        Default: 50

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/abs/1506.01497

    '''
    # calculate number of anchors that equal to product of length of anchor_ratio and length of anchor_scale
    num_anchors = len(anchor_ratio) * len(anchor_scale)
    parameters = locals()
    # get parameters of input, rpn, fast_rcnn layer
    input_parameters = _get_layer_options(input_layer_options, parameters)
    rpn_parameters = _get_layer_options(rpn_layer_options, parameters)
    fast_rcnn_parameters = _get_layer_options(fast_rcnn_options, parameters)
    inp = Input(**input_parameters, name='data')
    # backbone is VGG16 model
    conv1_1 = Conv2d(n_filters=64, width=3, height=3, stride=1, name='conv1_1')(inp)
    conv1_2 = Conv2d(n_filters=64, width=3, height=3, stride=1, name='conv1_2')(conv1_1)
    pool1 = Pooling(width=2, height=2, stride=2, pool='max', name='pool1')(conv1_2)

    conv2_1 = Conv2d(n_filters=128, width=3, height=3, stride=1, name='conv2_1')(pool1)
    conv2_2 = Conv2d(n_filters=128, width=3, height=3, stride=1, name='conv2_2')(conv2_1)
    pool2 = Pooling(width=2, height=2, stride=2, pool='max')(conv2_2)

    conv3_1 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_1')(pool2)
    conv3_2 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_2')(conv3_1)
    conv3_3 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_3')(conv3_2)
    pool3 = Pooling(width=2, height=2, stride=2, pool='max')(conv3_3)

    conv4_1 = Conv2d(n_filters=512, width=3, height=3, stride = 1, name = 'conv4_1')(pool3)
    conv4_2 = Conv2d(n_filters=512, width=3, height=3, stride = 1, name = 'conv4_2')(conv4_1)
    conv4_3 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv4_3')(conv4_2)
    pool4 = Pooling(width=2, height=2, stride=2, pool='max')(conv4_3)

    conv5_1 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_1')(pool4)
    conv5_2 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_2')(conv5_1)
    # feature of Conv5_3 is used to generate region proposals
    conv5_3 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_3')(conv5_2)
    # two convolutions build on top of conv5_3 and reduce feature map depth to 6*number_anchors
    rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(conv5_3)
    rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors),
                       name='rpn_score')(rpn_conv)
    # propose anchors, NMS, select anchors to train RPN, produce ROIs
    rp1 = RegionProposal(**rpn_parameters, name = 'rois')(rpn_score)
    # given ROIs, crop on conv5_3 and resize the feature to the same size
    roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                          spatial_scale=conv5_3.shape[0]/width,
                          name = 'roi_pooling')([conv5_3, rp1])
    # fully connect layer to extract the feature of ROIs
    fc6 = Dense(n=4096, act='relu', name='fc6')(roipool1)
    fc7 = Dense(n=4096, act='relu', name='fc7')(fc6)
    # classification tensor
    cls1 = Dense(n=n_classes+1, act='identity', name='cls_score')(fc7)
    # regression tensor(second stage bounding box regression)
    reg1 = Dense(n=(n_classes+1)*4, act='identity', name='bbox_pred')(fc7)
    # task layer receive cls1, reg1 and rp1(ground truth). Train the second stage.
    fr1 = FastRCNN(**fast_rcnn_parameters, class_number=n_classes, name='fastrcnn')([cls1, reg1, rp1])
    faster_rcnn = Model(conn, inp, fr1, model_table=model_table)
    faster_rcnn.compile()
    return faster_rcnn

