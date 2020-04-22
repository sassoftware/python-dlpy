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

from dlpy.model import Model
from dlpy.sequential import Sequential
from dlpy.layers import (Conv2d, Input, Pooling, BN, Conv2DTranspose, Concat, Res, Segmentation)
from .application_utils import get_layer_options, input_layer_options


def initial_block(inp):
    '''
    Defines the initial block of ENet

    Parameters
    ----------
    inp : class:`InputLayer`
    Input layer

    Returns
    -------
    :class:`Concat`
    '''
    x = Conv2d(13, 3, stride=2, padding=1, act='identity', include_bias=False)(inp)
    x_bn = BN(act='relu')(x)
    y = Pooling(2)(inp)
    merge = Concat()([x_bn, y])

    return merge


def downsampling_bottleneck(x, in_depth, out_depth, projection_ratio=4):
    '''
    Defines the down-sampling bottleneck of ENet

    Parameters
    ----------
    x : class:`Layer'
        Previous layer to this block
    in_depth : int
        Depth of the layer fed into this block
    out_depth : int
        Depth of the output layer of this block
    projection_ratio : int, optional
        Used to calculate the reduced_depth for intermediate convolution layers
        Default: 4

    Returns
    -------
    :class:`Res`
    '''

    reduced_depth = int(in_depth // projection_ratio)

    conv1 = Conv2d(reduced_depth, 3, stride=2, padding=1, act='identity', include_bias=False)(x)
    bn1 = BN(act='relu')(conv1)

    conv2 = Conv2d(reduced_depth, 3, stride=1, act='identity', include_bias=False)(bn1)
    bn2 = BN(act='relu')(conv2)

    conv3 = Conv2d(out_depth, 1, stride=1, act='identity', include_bias=False)(bn2)
    bn3 = BN(act='relu')(conv3)

    pool1 = Pooling(2, stride=2)(x)
    conv4 = Conv2d(out_depth, 1, stride=1, act='identity', include_bias=False)(pool1)
    bn4 = BN(act='relu')(conv4)

    res = Res()([bn3, bn4])

    return res


def regular_bottleneck(x, in_depth, out_depth, projection_ratio=4):
    '''
    Defines the regular bottleneck of ENet

    Parameters
    ----------
    x : class:`Layer'
       Previous layer to this block
    in_depth : int
       Depth of the layer fed into this block
    out_depth : int
       Depth of the output layer of this block
    projection_ratio : int, optional
       Used to calculate the reduced_depth for intermediate convolution layers
       Default: 4

    Returns
    -------
    :class:`Res`
    '''

    reduced_depth = int(in_depth // projection_ratio)

    conv1 = Conv2d(reduced_depth, 1, stride=1, act='identity', include_bias=False)(x)
    bn1 = BN(act='relu')(conv1)

    conv2 = Conv2d(reduced_depth, 3, stride=1, act='identity', include_bias=False)(bn1)
    bn2 = BN(act='relu')(conv2)

    conv3 = Conv2d(out_depth, 1, stride=1, act='identity', include_bias=False)(bn2)
    bn3 = BN(act='relu')(conv3)

    res = Res()([bn3, x])

    return res


def upsampling_bottleneck(x, in_depth, out_depth, projection_ratio=4):
    '''
    Defines the up-sampling bottleneck of ENet

    Parameters
    ----------
    x : class:`Layer'
       Previous layer to this block
    in_depth : int
       Depth of the layer fed into this block
    out_depth : int
       Depth of the output layer of this block
    projection_ratio : int, optional
       Used to calculate the reduced_depth for intermediate convolution layers
       Default: 4

    Returns
    -------
    :class:`BN`
    '''

    reduced_depth = int(in_depth // projection_ratio)

    conv1 = Conv2d(reduced_depth, 1, stride=1, act='identity', include_bias=False)(x)
    bn1 = BN(act='relu')(conv1)

    tconv1 = Conv2DTranspose(reduced_depth, 3, stride=2, padding=1, output_padding=1, act='identity',
                             include_bias=False)(bn1)
    bn2 = BN(act='relu')(tconv1)

    conv3 = Conv2d(out_depth, 1, stride=1, act='identity', include_bias=False)(bn2)
    bn3 = BN(act='relu')(conv3)

    return bn3


def ENet(conn, model_table='ENet', n_classes=2, n_channels=3, width=512, height=512, scale=1.0/255,
         norm_stds=None, offsets=None, random_mutation=None, init=None, random_flip=None, random_crop=None,
         output_image_type=None, output_image_prob=False):
    '''
    Generates a deep learning model with the E-Net architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
        Default: ENet
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 2
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 512
    height : int, optional
        Specifies the height of the input layer.
        Default: 512
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
    init : str
        Specifies the initialization scheme for convolution layers.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: None
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
    output_image_type: string, optional
        Specifies the output image type of this layer.
        possible values: [ WIDE, PNG, BASE64 ]
        default: WIDE
    output_image_prob: bool, options
        Does not include probabilities if doing classification (default).


    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/abs/1606.02147

    '''
    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)
    inp = Input(**input_parameters, name='InputLayer_1')

    # initial
    x = initial_block(inp)

    # stage one
    x = downsampling_bottleneck(x, 16, 64)
    for i in range(4):
        x = regular_bottleneck(x, 64, 64)

    # stage two
    x = downsampling_bottleneck(x, 64, 128)
    for i in range(2):
        x = regular_bottleneck(x, 128, 128)
        x = regular_bottleneck(x, 128, 128)

    # stage three
    for i in range(2):
        x = regular_bottleneck(x, 128, 128)
        x = regular_bottleneck(x, 128, 128)

    # stage four
    x = upsampling_bottleneck(x, 128, 64)
    for i in range(2):
        x = regular_bottleneck(x, 64, 64)

    # stage five
    x = upsampling_bottleneck(x, 64, 16)
    x = regular_bottleneck(x, 16, 16)

    x = upsampling_bottleneck(x, 16, 16)
    conv = Conv2d(n_classes, 3, act='relu')(x)

    seg = Segmentation(name='Segmentation_1', output_image_type=output_image_type,
                       output_image_prob=output_image_prob)(conv)

    model = Model(conn, inputs=inp, outputs=seg)
    model.compile()
    return model

