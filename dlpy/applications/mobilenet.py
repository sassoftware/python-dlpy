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

import os

from dlpy.model import Model
from dlpy.layers import Conv2d, BN, OutputLayer, Input, GroupConv2d, GlobalAveragePooling2D, Res, InputLayer
from .application_utils import get_layer_options, input_layer_options
from dlpy.utils import DLPyError
from dlpy.network import extract_input_layer, extract_output_layer, extract_conv_layer


def MobileNetV1(conn, model_table='MobileNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                random_flip=None, random_crop=None, random_mutation=None,
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
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
    input_parameters = get_layer_options(input_layer_options, parameters)
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
                random_flip=None, random_crop=None, random_mutation=None, alpha=1):
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
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
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
    input_parameters = get_layer_options(input_layer_options, parameters)
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


def MobileNetV2_ONNX(conn, model_file, n_classes=1000, width=224, height=224,
                     offsets=(255*0.406, 255*0.456, 255*0.485), norm_stds=(255*0.225, 255*0.224, 255*0.229),
                     random_flip=None, random_crop=None, random_mutation=None, include_top=False):
    """
    Generates a deep learning model with the MobileNetV2_ONNX architecture.
    The model architecture and pre-trained weights is generated from MobileNetV2 ONNX trained on ImageNet dataset.
    The model file and the weights file can be downloaded from https://support.sas.com/documentation/prod-p/vdmml/zip/.
    To learn more information about the model and pre-processing.
    Please go to the websites: https://github.com/onnx/models/tree/master/vision/classification/mobilenet.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_file : string
        Specifies the absolute server-side path of the model table file.
        The model table file can be downloaded from https://support.sas.com/documentation/prod-p/vdmml/zip/.
    n_classes : int, optional
        Specifies the number of classes.
        Default: 1000
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        The channel order is BGR.
        Default: (255*0.406, 255*0.456, 255*0.485)
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
        The channel order is BGR.
        Default: (255*0.225, 255*0.224, 255*0.229)
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
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers)
        Default: False

    """
    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)

    # load model and model weights
    model = Model.from_sashdat(conn, path = model_file)
    # check if a user points to a correct model.
    if model.summary.shape[0] != 120:
        raise DLPyError("The model file doesn't point to a valid MobileNetV2_ONNX model. "
                        "Please check the SASHDAT file.")
    # extract input layer config
    model_table_df = conn.CASTable(**model.model_table).to_frame()
    input_layer_df = model_table_df[model_table_df['_DLLayerID_'] == 0]
    input_layer = extract_input_layer(input_layer_df)
    input_layer_config = input_layer.config
    # update input layer config
    input_layer_config.update(input_parameters)
    # update the layer list
    model.layers[0] = InputLayer(**input_layer_config, name=model.layers[0].name)

    # warning if model weights doesn't exist
    if not conn.tableexists(model.model_weights.name).exists:
        weights_file_path = os.path.join(os.path.dirname(model_file), model.model_name + '_weights.sashdat')
        print('WARNING: Model weights is not attached '
              'since system cannot find a weights file located at {}'.format(weights_file_path))

    if include_top:
        if n_classes != 1000:
            raise DLPyError("If include_top is enabled, n_classes has to be 1000.")
    else:
        # since the output layer is non fully connected layer,
        # we need to modify the convolution right before the output. The number of filter is set to n_classes.
        conv_layer_df = model_table_df[model_table_df['_DLLayerID_'] == 118]
        conv_layer = extract_conv_layer(conv_layer_df)
        conv_layer_config = conv_layer.config
        # update input layer config
        conv_layer_config.update({'n_filters': n_classes})
        # update the layer list
        model.layers[-2] = Conv2d(**conv_layer_config,
                                  name=model.layers[-2].name, src_layers=model.layers[-3])

        # overwrite n_classes in output layer
        out_layer_df = model_table_df[model_table_df['_DLLayerID_'] == 119]
        out_layer = extract_output_layer(out_layer_df)
        out_layer_config = out_layer.config
        # update input layer config
        out_layer_config.update({'n': n_classes})
        # update the layer list
        model.layers[-1] = OutputLayer(**out_layer_config,
                                       name = model.layers[-1].name, src_layers=model.layers[-2])

        # remove top weights
        model.model_weights.append_where('_LayerID_<118')
        model._retrieve_('table.partition', table=model.model_weights,
                         casout=dict(replace=True, name=model.model_weights.name))
        model.set_weights(model.model_weights.name)
    # recompile the whole network according to the new layer list
    model.compile()
    return model

