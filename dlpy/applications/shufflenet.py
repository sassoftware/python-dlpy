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


from dlpy.layers import (Conv2d, BN, Pooling, Concat, OutputLayer, GlobalAveragePooling2D, GroupConv2d,
                         ChannelShuffle, Res, Input)
from dlpy.utils import DLPyError
from dlpy.model import Model
from .application_utils import get_layer_options, input_layer_options


def ShuffleNetV1(conn, model_table='ShuffleNetV1', n_classes=1000, n_channels=3, width=224, height=224,
                 norm_stds=(255*0.229, 255*0.224, 255*0.225), offsets=(255*0.485, 255*0.456, 255*0.406),
                 random_flip=None, random_crop=None, random_mutation=None, scale_factor=1.0,
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
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
    input_parameters = get_layer_options(input_layer_options, parameters)
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