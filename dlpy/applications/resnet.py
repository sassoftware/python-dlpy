#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import warnings

from dlpy.sequential import Sequential
from dlpy.model import Model
from dlpy.layers import InputLayer, Conv2d, BN, Pooling, GlobalAveragePooling2D, OutputLayer, Reshape
from dlpy.blocks import ResBlockBN, ResBlock_Caffe
from dlpy.utils import DLPyError, check_layer_class
from dlpy.caffe_models import (model_resnet50, model_resnet101, model_resnet152)
from .application_utils import get_layer_options, input_layer_options


def ResNet18_SAS(conn, model_table='RESNET18_SAS', batch_norm_first=True, n_classes=1000, n_channels=3, width=224,
                 height=224, scale=1, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                 random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                   height=224, scale=1, random_flip=None, random_crop=None, offsets=None,
                   random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                 batch_norm_first=True, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                 random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                   batch_norm_first=False, random_flip=None, random_crop=None, offsets=None,
                   random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: None
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                 batch_norm_first=True, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                 random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                   batch_norm_first=False, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                   pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
                   random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

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

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

        # when a RNN model is built to consume the input data, it automatically flattens the input tensor
        # to a one-dimension vector. The reshape layer is required to reshape the tensor to the original definition.
        # This feature of mixing CNN layers with a RNN model is supported in VDMML 8.5.
        # This option could be also used to reshape the input tensor.
        if reshape_after_input:
            model.add(reshape_after_input)

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
                                                  width=width, height=height, random_crop=random_crop, offsets=offsets,
                                                  random_flip=random_flip, random_mutation=random_mutation,
                                                  reshape_after_input=reshape_after_input)

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
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc1000',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['pool5'])

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<125'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model = Model.from_table(conn.CASTable(model_table))
            return model


def ResNet101_SAS(conn, model_table='RESNET101_SAS',  n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                  batch_norm_first=True, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                  random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                    batch_norm_first=False, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
                    random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

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

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

        # add reshape when specified
        if reshape_after_input:
            model.add(reshape_after_input)

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
                                                     offsets=offsets,
                                                     random_flip=random_flip, random_mutation=random_mutation,
                                                     reshape_after_input=reshape_after_input)

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
                  batch_norm_first=True, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                  random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1512.03385.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
                    batch_norm_first=False, random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                    pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
                    random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

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

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    # get all the parms passed in
    parameters = locals()

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

        # add reshape when specified
        if reshape_after_input:
            model.add(reshape_after_input)

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
                                                     offsets=offsets,
                                                     random_flip=random_flip, random_mutation=random_mutation,
                                                     reshape_after_input=reshape_after_input)

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
                n_channels=3, width=32, height=32, scale=1, random_flip=None, random_crop=None,
                offsets=(103.939, 116.779, 123.68),
                random_mutation=None, reshape_after_input=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1605.07146.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    in_filters = 16

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    # add reshape when specified
    if reshape_after_input:
        model.add(reshape_after_input)

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
