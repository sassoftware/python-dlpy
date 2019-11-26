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

from dlpy.sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, BN, Pooling, Concat, OutputLayer, GlobalAveragePooling2D
from dlpy.blocks import DenseNetBlock
from .application_utils import get_layer_options, input_layer_options
from dlpy.model import Model
from dlpy.utils import DLPyError
from dlpy.network import extract_input_layer, extract_output_layer, extract_conv_layer


def DenseNet(conn, model_table='DenseNet', n_classes=None, conv_channel=16, growth_rate=12, n_blocks=4,
             n_cells=4, n_channels=3, width=32, height=32, scale=1, random_flip=None, random_crop=None,
             offsets=(85, 111, 139), random_mutation=None):
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
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (85, 111, 139)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1608.06993.pdf

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    channel_in = conv_channel  # number of channel of transition conv layer

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

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
                random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68), random_mutation=None):
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1608.06993.pdf

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    n_blocks = len(n_cells)

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

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


def DenseNet121_ONNX(conn, model_file, n_classes=1000, width=224, height=224,
                     offsets=(255*0.406, 255*0.456, 255*0.485), norm_stds=(255*0.225, 255*0.224, 255*0.229),
                     random_flip=None, random_crop=None, random_mutation=None, include_top=False):
    """
    Generates a deep learning model with the DenseNet121_ONNX architecture.
    The model architecture and pre-trained weights is generated from DenseNet121 ONNX trained on ImageNet dataset.
    The model file and the weights file can be downloaded from https://support.sas.com/documentation/prod-p/vdmml/zip/.
    To learn more information about the model and pre-processing.
    Please go to the websites: https://github.com/onnx/models/tree/master/vision/classification/densenet-121.

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
    if model.summary.shape[0] != 307:
        raise DLPyError("The model file doesn't point to a valid DenseNet121_ONNX model. "
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
        conv_layer_df = model_table_df[model_table_df['_DLLayerID_'] == 305]
        conv_layer = extract_conv_layer(conv_layer_df)
        conv_layer_config = conv_layer.config
        # update input layer config
        conv_layer_config.update({'n_filters': n_classes})
        # update the layer list
        model.layers[-2] = Conv2d(**conv_layer_config,
                                  name=model.layers[-2].name, src_layers=model.layers[-3])

        # overwrite n_classes in output layer
        out_layer_df = model_table_df[model_table_df['_DLLayerID_'] == 306]
        out_layer = extract_output_layer(out_layer_df)
        out_layer_config = out_layer.config
        # update input layer config
        out_layer_config.update({'n': n_classes})
        # update the layer list
        model.layers[-1] = OutputLayer(**out_layer_config,
                                       name = model.layers[-1].name, src_layers=model.layers[-2])

        # remove top weights
        model.model_weights.append_where('_LayerID_<305')
        model._retrieve_('table.partition', table=model.model_weights,
                         casout=dict(replace=True, name=model.model_weights.name))
        model.set_weights(model.model_weights.name)
    # recompile the whole network according to the new layer list
    model.compile()
    return model

