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
from dlpy.layers import (Conv2d, Input, Pooling, BN, Conv2DTranspose, Concat, Segmentation)
from .application_utils import get_layer_options, input_layer_options


def UNet(conn, model_table='UNet', n_classes = 2, n_channels=1, width=256, height=256, scale=1.0/255,
         norm_stds=None, offsets=None, random_mutation=None, init=None, bn_after_convolutions=False,
         random_flip=None, random_crop=None, output_image_type=None, output_image_prob=False):
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
    init : str
        Specifies the initialization scheme for convolution layers.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: None
    bn_after_convolutions : Boolean
        If set to True, a batch normalization layer is added after each convolution layer.
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
    https://arxiv.org/pdf/1505.04597

    '''
    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)
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
