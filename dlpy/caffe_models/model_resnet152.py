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
from ..utils import input_table_check


def ResNet152_Model(s, model_table='RESNET152', n_channels=3, width=224, height=224,
                    random_crop=None, offsets=None):
    '''
    ResNet152 model definition

    Parameters
    ----------
    s : CAS
        Specifies the CAS connection object
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels of the input layer
        Default: 3
    width : int, optional
        Specifies the width of the input layer
        Default: 224
    height : int, optional
        Specifies the height of the input layer
        Default: 224
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters.deepLearn. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default	: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.deepLearn.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    None
        A CAS table defining the model is created

    '''
    model_table_opts = input_table_check(model_table)

    # quick error-checking and default setting
    if random_crop is None:
        random_crop = 'none'
    elif random_crop.lower() not in ['none', 'unique']:
        raise ValueError('random_crop can only be "none" or "unique"')

    if offsets is None:
        offsets = [103.939, 116.779, 123.68]

    # instantiate model
    s.deepLearn.buildModel(model=dict(replace=True, **model_table_opts), type='CNN')

    # input layer
    s.deepLearn.addLayer(model=model_table_opts, name='data',
                         layer=dict(type='input', nchannels=n_channels, width=width, height=height,
                                    randomcrop=random_crop, offsets=offsets))

    # -------------------- Layer 1 ----------------------

    # conv1 layer: 64 channels, 7x7 conv, stride=2; output = 112 x 112 */
    s.deepLearn.addLayer(model=model_table_opts, name='conv1',
                         layer=dict(type='convolution', nFilters=64, width=7, height=7,
                                    stride=2, act='identity'),
                         srcLayers=['data'])

    # conv1 batch norm layer: 64 channels, output = 112 x 112 */
    s.deepLearn.addLayer(model=model_table_opts, name='bn_conv1',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv1'])

    # pool1 layer: 64 channels, 3x3 pooling, output = 56 x 56 */
    s.deepLearn.addLayer(model=model_table_opts, name='pool1',
                         layer=dict(type='pooling', width=3, height=3, stride=2, pool='max'),
                         srcLayers=['bn_conv1'])

    # ------------------- Residual Layer 2A -----------------------

    # res2a_branch1 layer: 256 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2a_branch1',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['pool1'])

    # res2a_branch1 batch norm layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res2a_branch1'])

    # res2a_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2a_branch2a',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['pool1'])

    # res2a_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2a_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2a_branch2a'])

    # res2a_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2a_branch2b',
                         layer=dict(type='convolution', nFilters=64, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2a_branch2a'])

    # res2a_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2a_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2a_branch2b'])

    # res2a_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2a_branch2c',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2a_branch2b'])

    # res2a_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2a_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res2a_branch2c'])

    # res2a residual layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2a',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn2a_branch2c', 'bn2a_branch1'])

    # ------------------- Residual Layer 2B -----------------------

    # res2b_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2b_branch2a',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res2a'])

    # res2b_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2b_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2b_branch2a'])

    # res2b_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2b_branch2b',
                         layer=dict(type='convolution', nFilters=64, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2b_branch2a'])

    # res2b_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2b_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2b_branch2b'])

    # res2b_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2b_branch2c',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2b_branch2b'])

    # res2b_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2b_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res2b_branch2c'])

    # res2b residual layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2b',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn2b_branch2c', 'res2a'])

    # ------------------- Residual Layer 2C -----------------------

    # res2c_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2c_branch2a',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res2b'])

    # res2c_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2c_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2c_branch2a'])

    # res2c_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2c_branch2b',
                         layer=dict(type='convolution', nFilters=64, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2c_branch2a'])

    # res2c_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2c_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res2c_branch2b'])

    # res2c_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2c_branch2c',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn2c_branch2b'])

    # res2c_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='bn2c_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res2c_branch2c'])

    # res2c residual layer: 256 channels, output = 56 x 56
    s.deepLearn.addLayer(model=model_table_opts, name='res2c',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn2c_branch2c', 'res2b'])

    # ------------- Layer 3A --------------------

    # res3a_branch1 layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3a_branch1',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res2c'])

    # res3a_branch1 batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3a_branch1'])

    # res3a_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3a_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res2c'])

    # res3a_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3a_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3a_branch2a'])

    # res3a_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3a_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3a_branch2a'])

    # res3a_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3a_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3a_branch2b'])

    # res3a_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3a_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3a_branch2b'])

    # res3a_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3a_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3a_branch2c'])

    # res3a residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3a',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3a_branch2c', 'bn3a_branch1'])

    # ------------------- Residual Layer 3B1 -----------------------

    # res3b1_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b1_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3a'])

    # res3b1_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b1_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b1_branch2a'])

    # res3b1_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b1_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b1_branch2a'])

    # res3b1_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b1_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b1_branch2b'])

    # res3b1_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b1_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b1_branch2b'])

    # res3b1_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b1_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b1_branch2c'])

    # res3b1 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b1',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b1_branch2c', 'res3a'])

    # ------------------- Residual Layer 3B2 -----------------------

    # res3b2_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b2_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b1'])

    # res3b2_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b2_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b2_branch2a'])

    # res3b2_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b2_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b2_branch2a'])

    # res3b2_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b2_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b2_branch2b'])

    # res3b2_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b2_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b2_branch2b'])

    # res3b2_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b2_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b2_branch2c'])

    # res3b2 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b2',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b2_branch2c', 'res3b1'])

    # ------------------- Residual Layer 3B3 -----------------------

    # res3b3_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b3_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b2'])

    # res3b3_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b3_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b3_branch2a'])

    # res3b3_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b3_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b3_branch2a'])

    # res3b3_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b3_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b3_branch2b'])

    # res3b3_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b3_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b3_branch2b'])

    # res3b3_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b3_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b3_branch2c'])

    # res3b3 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b3',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b3_branch2c', 'res3b2'])

    # ------------------- Residual Layer 3B4 -----------------------

    # res3b4_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b4_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b3'])

    # res3b4_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b4_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b4_branch2a'])

    # res3b4_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b4_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b4_branch2a'])

    # res3b4_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b4_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b4_branch2b'])

    # res3b4_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b4_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b4_branch2b'])

    # res3b4_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b4_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b4_branch2c'])

    # res3b4 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b4',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b4_branch2c', 'res3b3'])

    # ------------------- Residual Layer 3B5 -----------------------

    # res3b5_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b5_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b4'])

    # res3b5_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b5_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b5_branch2a'])

    # res3b5_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b5_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b5_branch2a'])

    # res3b5_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b5_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b5_branch2b'])

    # res3b5_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b5_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b5_branch2b'])

    # res3b5_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b5_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b5_branch2c'])

    # res3b5 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b5',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b5_branch2c', 'res3b4'])

    # ------------------- Residual Layer 3B6 -----------------------

    # res3b6_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b6_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b5'])

    # res3b6_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b6_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b6_branch2a'])

    # res3b6_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b6_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b6_branch2a'])

    # res3b6_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b6_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b6_branch2b'])

    # res3b6_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b6_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b6_branch2b'])

    # res3b6_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b6_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b6_branch2c'])

    # res3b6 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b6',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b6_branch2c', 'res3b5'])

    # ------------------- Residual Layer 3B7 -----------------------

    # res3b7_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b7_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b6'])

    # res3b7_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b7_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b7_branch2a'])

    # res3b7_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b7_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b7_branch2a'])

    # res3b7_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b7_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b7_branch2b'])

    # res3b7_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b7_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b7_branch2b'])

    # res3b7_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b7_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b7_branch2c'])

    # res3b7 residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b7',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b7_branch2c', 'res3b6'])

    # ------------- Layer 4A --------------------

    # res4a_branch1 layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch1',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res3b7'])

    # res4a_branch1 batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4a_branch1'])

    # res4a_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res3b7'])

    # res4a_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4a_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4a_branch2a'])

    # res4a_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4a_branch2a'])

    # res4a_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4a_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4a_branch2b'])

    # res4a_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4a_branch2b'])

    # res4a_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4a_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4a_branch2c'])

    # res4a residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4a_branch2c', 'bn4a_branch1'])

    # ------------------- Residual Layer 4B1 -----------------------

    # res4b1_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b1_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4a'])

    # res4b1_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b1_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b1_branch2a'])

    # res4b1_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b1_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b1_branch2a'])

    # res4b1_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b1_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b1_branch2b'])

    # res4b1_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b1_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b1_branch2b'])

    # res4b1_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b1_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b1_branch2c'])

    # res4b1 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b1',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b1_branch2c', 'res4a'])

    # ------------------- Residual Layer 4B2 -----------------------

    # res4b2_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b2_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b1'])

    # res4b2_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b2_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b2_branch2a'])

    # res4b2_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b2_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b2_branch2a'])

    # res4b2_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b2_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b2_branch2b'])

    # res4b2_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b2_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b2_branch2b'])

    # res4b2_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b2_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b2_branch2c'])

    # res4b2 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b2',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b2_branch2c', 'res4b1'])

    # ------------------- Residual Layer 4B3 -----------------------

    # res4b3_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b3_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b2'])

    # res4b3_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b3_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b3_branch2a'])

    # res4b3_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b3_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b3_branch2a'])

    # res4b3_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b3_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b3_branch2b'])

    # res4b3_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b3_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b3_branch2b'])

    # res4b3_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b3_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b3_branch2c'])

    # res4b3 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b3',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b3_branch2c', 'res4b2'])

    # ------------------- Residual Layer 4B4 ----------------------- */

    # res4b4_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b4_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b3'])

    # res4b4_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b4_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b4_branch2a'])

    # res4b4_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b4_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b4_branch2a'])

    # res4b4_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b4_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b4_branch2b'])

    # res4b4_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b4_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b4_branch2b'])

    # res4b4_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b4_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b4_branch2c'])

    # res4b4 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b4',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b4_branch2c', 'res4b3'])

    # ------------------- Residual Layer 4B5 -----------------------

    # res4b5_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b5_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b4'])

    # res4b5_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b5_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b5_branch2a'])

    # res4b5_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b5_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b5_branch2a'])

    # res4b5_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b5_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b5_branch2b'])

    # res4b5_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b5_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b5_branch2b'])

    # res4b5_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b5_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b5_branch2c'])

    # res4b5 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b5',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b5_branch2c', 'res4b4'])

    # ------------------- Residual Layer 4B6 -----------------------

    # res4b6_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b6_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b5'])

    # res4b6_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b6_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b6_branch2a'])

    # res4b6_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b6_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b6_branch2a'])

    # res4b6_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b6_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b6_branch2b'])

    # res4b6_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b6_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b6_branch2b'])

    # res4b6_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b6_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b6_branch2c'])

    # res4b6 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b6',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b6_branch2c', 'res4b5'])

    # ------------------- Residual Layer 4B7 -----------------------

    # res4b7_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b7_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b6'])

    # res4b7_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b7_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b7_branch2a'])

    # res4b7_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b7_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b7_branch2a'])

    # res4b7_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b7_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b7_branch2b'])

    # res4b7_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b7_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b7_branch2b'])

    # res4b7_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b7_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b7_branch2c'])

    # res4b7 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b7',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b7_branch2c', 'res4b6'])

    # ------------------- Residual Layer 4B8 -----------------------

    # res4b8_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b8_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b7'])

    # res4b8_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b8_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b8_branch2a'])

    # res4b8_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b8_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b8_branch2a'])

    # res4b8_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b8_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b8_branch2b'])

    # res4b8_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b8_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b8_branch2b'])

    # res4b8_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b8_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b8_branch2c'])

    # res4b8 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b8',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b8_branch2c', 'res4b7'])

    # ------------------- Residual Layer 4B9 -----------------------

    # res4b9_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b9_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b8'])

    # res4b9_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b9_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b9_branch2a'])

    # res4b9_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b9_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b9_branch2a'])

    # res4b9_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b9_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b9_branch2b'])

    # res4b9_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b9_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b9_branch2b'])

    # res4b9_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b9_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b9_branch2c'])

    # res4b9 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b9',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b9_branch2c', 'res4b8'])

    # ------------------- Residual Layer 4B10 -----------------------

    # res4b10_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b10_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b9'])

    # res4b10_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b10_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b10_branch2a'])

    # res4b10_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b10_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b10_branch2a'])

    # res4b10_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b10_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b10_branch2b'])

    # res4b10_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b10_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b10_branch2b'])

    # res4b10_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b10_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b10_branch2c'])

    # res4b10 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b10',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b10_branch2c', 'res4b9'])

    # ------------------- Residual Layer 4B11 -----------------------

    # res4b11_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b11_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b10'])

    # res4b11_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b11_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b11_branch2a'])

    # res4b11_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b11_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b11_branch2a'])

    # res4b11_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b11_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b11_branch2b'])

    # res4b11_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b11_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b11_branch2b'])

    # res4b11_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b11_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b11_branch2c'])

    # res4b11 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b11',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b11_branch2c', 'res4b10'])

    # ------------------- Residual Layer 4B12 -----------------------

    # res4b12_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b12_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b11'])

    # res4b12_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b12_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b12_branch2a'])

    # res4b12_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b12_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b12_branch2a'])

    # res4b12_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b12_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b12_branch2b'])

    # res4b12_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b12_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b12_branch2b'])

    # res4b12_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b12_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b12_branch2c'])

    # res4b12 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b12',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b12_branch2c', 'res4b11'])

    # ------------------- Residual Layer 4B13 -----------------------

    # res4b13_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b13_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b12'])

    # res4b13_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b13_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b13_branch2a'])

    # res4b13_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b13_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b13_branch2a'])

    # res4b13_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b13_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b13_branch2b'])

    # res4b13_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b13_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b13_branch2b'])

    # res4b13_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b13_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b13_branch2c'])

    # res4b13 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b13',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b13_branch2c', 'res4b12'])

    # ------------------- Residual Layer 4B14 -----------------------

    # res4b14_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b14_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b13'])

    # res4b14_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b14_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b14_branch2a'])

    # res4b14_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b14_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b14_branch2a'])

    # res4b14_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b14_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b14_branch2b'])

    # res4b14_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b14_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b14_branch2b'])

    # res4b14_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b14_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b14_branch2c'])

    # res4b14 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b14',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b14_branch2c', 'res4b13'])

    # ------------------- Residual Layer 4B15 -----------------------

    # res4b15_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b15_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b14'])

    # res4b15_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b15_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b15_branch2a'])

    # res4b15_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b15_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b15_branch2a'])

    # res4b15_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b15_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b15_branch2b'])

    # res4b15_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b15_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b15_branch2b'])

    # res4b15_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b15_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b15_branch2c'])

    # res4b15 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b15',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b15_branch2c', 'res4b14'])

    # ------------------- Residual Layer 4B16 -----------------------

    # res4b16_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b16_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b15'])

    # res4b16_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b16_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b16_branch2a'])

    # res4b16_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b16_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b16_branch2a'])

    # res4b16_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b16_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b16_branch2b'])

    # res4b16_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b16_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b16_branch2b'])

    # res4b16_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b16_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b16_branch2c'])

    # res4b16 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b16',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b16_branch2c', 'res4b15'])

    # ------------------- Residual Layer 4B17 -----------------------

    # res4b17_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b17_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b16'])

    # res4b17_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b17_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b17_branch2a'])

    # res4b17_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b17_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b17_branch2a'])

    # res4b17_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b17_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b17_branch2b'])

    # res4b17_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b17_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b17_branch2b'])

    # res4b17_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b17_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b17_branch2c'])

    # res4b17 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b17',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b17_branch2c', 'res4b16'])

    # ------------------- Residual Layer 4B18 -----------------------

    # res4b18_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b18_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b17'])

    # res4b18_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b18_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b18_branch2a'])

    # res4b18_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b18_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b18_branch2a'])

    # res4b18_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b18_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b18_branch2b'])

    # res4b18_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b18_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b18_branch2b'])

    # res4b18_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b18_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b18_branch2c'])

    # res4b18 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b18',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b18_branch2c', 'res4b17'])

    # ------------------- Residual Layer 4B19 -----------------------

    # res4b19_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b19_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b18'])

    # res4b19_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b19_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b19_branch2a'])

    # res4b19_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b19_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b19_branch2a'])

    # res4b19_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b19_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b19_branch2b'])

    # res4b19_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b19_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b19_branch2b'])

    # res4b19_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b19_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b19_branch2c'])

    # res4b19 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b19',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b19_branch2c', 'res4b18'])

    # ------------------- Residual Layer 4B20 -----------------------

    # res4b20_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b20_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b19'])

    # res4b20_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b20_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b20_branch2a'])

    # res4b20_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b20_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b20_branch2a'])

    # res4b20_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b20_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b20_branch2b'])

    # res4b20_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b20_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b20_branch2b'])

    # res4b20_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b20_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b20_branch2c'])

    # res4b20 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b20',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b20_branch2c', 'res4b19'])

    # ------------------- Residual Layer 4B21 -----------------------

    # res4b21_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b21_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b20'])

    # res4b21_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b21_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b21_branch2a'])

    # res4b21_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b21_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b21_branch2a'])

    # res4b21_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b21_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b21_branch2b'])

    # res4b21_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b21_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b21_branch2b'])

    # res4b21_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b21_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b21_branch2c'])

    # res4b21 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b21',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b21_branch2c', 'res4b20'])

    # ------------------- Residual Layer 4B22 -----------------------

    # res4b22_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b22_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b21'])

    # res4b22_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b22_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b22_branch2a'])

    # res4b22_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b22_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b22_branch2a'])

    # res4b22_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b22_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b22_branch2b'])

    # res4b22_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b22_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b22_branch2b'])

    # res4b22_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b22_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b22_branch2c'])

    # res4b22 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b22',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b22_branch2c', 'res4b21'])

    # ------------------- Residual Layer 4B23 -----------------------

    # res4b23_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b23_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b22'])

    # res4b23_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b23_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b23_branch2a'])

    # res4b23_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b23_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b23_branch2a'])

    # res4b23_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b23_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b23_branch2b'])

    # res4b23_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b23_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b23_branch2b'])

    # res4b23_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b23_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b23_branch2c'])

    # res4b23 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b23',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b23_branch2c', 'res4b22'])

    # ------------------- Residual Layer 4B24 -----------------------

    # res4b24_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b24_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b23'])

    # res4b24_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b24_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b24_branch2a'])

    # res4b24_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b24_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b24_branch2a'])

    # res4b24_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b24_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b24_branch2b'])

    # res4b24_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b24_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b24_branch2b'])

    # res4b24_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b24_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b24_branch2c'])

    # res4b24 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b24',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b24_branch2c', 'res4b23'])

    # ------------------- Residual Layer 4B25 -----------------------

    # res4b25_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b25_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b24'])

    # res4b25_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b25_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b25_branch2a'])

    # res4b25_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b25_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b25_branch2a'])

    # res4b25_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b25_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b25_branch2b'])

    # res4b25_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b25_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b25_branch2b'])

    # res4b25_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b25_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b25_branch2c'])

    # res4b25 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b25',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b25_branch2c', 'res4b24'])

    # ------------------- Residual Layer 4B26 -----------------------

    # res4b26_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b26_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b25'])

    # res4b26_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b26_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b26_branch2a'])

    # res4b26_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b26_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b26_branch2a'])

    # res4b26_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b26_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b26_branch2b'])

    # res4b26_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b26_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b26_branch2b'])

    # res4b26_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b26_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b26_branch2c'])

    # res4b26 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b26',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b26_branch2c', 'res4b25'])

    # ------------------- Residual Layer 4B27 -----------------------

    # res4b27_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b27_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b26'])

    # res4b27_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b27_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b27_branch2a'])

    # res4b27_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b27_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b27_branch2a'])

    # res4b27_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b27_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b27_branch2b'])

    # res4b27_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b27_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b27_branch2b'])

    # res4b27_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b27_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b27_branch2c'])

    # res4b27 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b27',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b27_branch2c', 'res4b26'])

    # ------------------- Residual Layer 4B28 -----------------------

    # res4b28_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b28_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b27'])

    # res4b28_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b28_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b28_branch2a'])

    # res4b28_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b28_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b28_branch2a'])

    # res4b28_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b28_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b28_branch2b'])

    # res4b28_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b28_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b28_branch2b'])

    # res4b28_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b28_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b28_branch2c'])

    # res4b28 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b28',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b28_branch2c', 'res4b27'])

    # ------------------- Residual Layer 4B29 -----------------------

    # res4b29_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b29_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b28'])

    # res4b29_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b29_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b29_branch2a'])

    # res4b29_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b29_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b29_branch2a'])

    # res4b29_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b29_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b29_branch2b'])

    # res4b29_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b29_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b29_branch2b'])

    # res4b29_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b29_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b29_branch2c'])

    # res4b29 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b29',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b29_branch2c', 'res4b28'])

    # ------------------- Residual Layer 4B30 -----------------------

    # res4b30_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b30_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b29'])

    # res4b30_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b30_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b30_branch2a'])

    # res4b30_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b30_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b30_branch2a'])

    # res4b30_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b30_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b30_branch2b'])

    # res4b30_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b30_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b30_branch2b'])

    # res4b30_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b30_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b30_branch2c'])

    # res4b30 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b30',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b30_branch2c', 'res4b29'])

    # ------------------- Residual Layer 4B31 -----------------------

    # res4b31_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b31_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b30'])

    # res4b31_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b31_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b31_branch2a'])

    # res4b31_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b31_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b31_branch2a'])

    # res4b31_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b31_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b31_branch2b'])

    # res4b31_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b31_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b31_branch2b'])

    # res4b31_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b31_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b31_branch2c'])

    # res4b31 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b31',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b31_branch2c', 'res4b30'])

    # ------------------- Residual Layer 4B32 -----------------------

    # res4b32_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b32_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b31'])

    # res4b32_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b32_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b32_branch2a'])

    # res4b32_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b32_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b32_branch2a'])

    # res4b32_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b32_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b32_branch2b'])

    # res4b32_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b32_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b32_branch2b'])

    # res4b32_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b32_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b32_branch2c'])

    # res4b32 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b32',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b32_branch2c', 'res4b31'])

    # ------------------- Residual Layer 4B33 -----------------------

    # res4b33_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b33_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b32'])

    # res4b33_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b33_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b33_branch2a'])

    # res4b33_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b33_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b33_branch2a'])

    # res4b33_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b33_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b33_branch2b'])

    # res4b33_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b33_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b33_branch2b'])

    # res4b33_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b33_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b33_branch2c'])

    # res4b33 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b33',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b33_branch2c', 'res4b32'])

    # ------------------- Residual Layer 4B34 -----------------------

    # res4b34_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b34_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b33'])

    # res4b34_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b34_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b34_branch2a'])

    # res4b34_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b34_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b34_branch2a'])

    # res4b34_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b34_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b34_branch2b'])

    # res4b34_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b34_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b34_branch2b'])

    # res4b34_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b34_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b34_branch2c'])

    # res4b34 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b34',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b34_branch2c', 'res4b33'])

    # ------------------- Residual Layer 4B35 -----------------------

    # res4b35_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b35_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b34'])

    # res4b35_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b35_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b35_branch2a'])

    # res4b35_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b35_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b35_branch2a'])

    # res4b35_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b35_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b35_branch2b'])

    # res4b35_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b35_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b35_branch2b'])

    # res4b35_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b35_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b35_branch2c'])

    # res4b35 residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b35',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b35_branch2c', 'res4b34'])

    # ------------- Layer 5A -------------------- */

    # res5a_branch1 layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch1',
                         layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res4b35'])

    # res5a_branch1 batch norm layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res5a_branch1'])

    # res5a_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch2a',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res4b35'])

    # res5a_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5a_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5a_branch2a'])

    # res5a_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch2b',
                         layer=dict(type='convolution', nFilters=512, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5a_branch2a'])

    # res5a_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5a_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5a_branch2b'])

    # res5a_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch2c',
                         layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5a_branch2b'])

    # res5a_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5a_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res5a_branch2c'])

    # res5a residual layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn5a_branch2c', 'bn5a_branch1'])

    # ------------------- Residual Layer 5B -----------------------

    # res5b_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5b_branch2a',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res5a'])

    # res5b_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5b_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5b_branch2a'])

    # res5b_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5b_branch2b',
                         layer=dict(type='convolution', nFilters=512, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5b_branch2a'])

    # res5b_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5b_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5b_branch2b'])

    # res5b_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5b_branch2c',
                         layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5b_branch2b'])

    # res5b_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5b_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res5b_branch2c'])

    # res5b residual layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5b',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn5b_branch2c', 'res5a'])

    # ------------------- Residual Layer 5C -----------------------

    # res5c_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5c_branch2a',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res5b'])

    # res5c_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5c_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5c_branch2a'])

    # res5c_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5c_branch2b',
                         layer=dict(type='convolution', nFilters=512, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5c_branch2a'])

    # res5c_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5c_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res5c_branch2b'])

    # res5c_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5c_branch2c',
                         layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn5c_branch2b'])

    # res5c_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5c_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res5c_branch2c'])

    # res5c residual layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5c',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn5c_branch2c', 'res5b'])

    # ------------------- final layers ----------------------

    # pool5 layer: 2048 channels, 7x7 pooling, output = 1 x 1
    kernel_width = width // 2 // 2 // 2 // 2 // 2
    kernel_height = height // 2 // 2 // 2 // 2 // 2
    stride = kernel_width

    s.deepLearn.addLayer(model=model_table_opts, name='pool5',
                         layer=dict(type='pooling', width=kernel_width,
                                    height=kernel_height, stride=stride, pool='mean'),
                         srcLayers=['res5c'])

    # fc1000 output layer: 1000 neurons */
    s.deepLearn.addLayer(model=model_table_opts, name='fc1000',
                         layer=dict(type='output', n=1000, act='softmax'),
                         srcLayers=['pool5'])

    return s.CASTable(**model_table_opts)
