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


def ResNet50_Model(s, model_table='RESNET50', n_channels=3, width=224, height=224,
                   random_crop=None, offsets=None):
    '''
    ResNet50 model definition

    Parameters
    ----------
    s : CAS
        Specifies the CAS connection object.
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
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
        CAS table defining the model is created

    '''

    model_table_opts = input_table_check(model_table)

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

    # ------------------- Residual Layer 3B -----------------------

    # res3b_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3a'])

    # res3b_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b_branch2a'])

    # res3b_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b_branch2a'])

    # res3b_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3b_branch2b'])

    # res3b_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3b_branch2b'])

    # res3b_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3b_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3b_branch2c'])

    # res3b residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3b',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3b_branch2c', 'res3a'])

    # ------------------- Residual Layer 3C -----------------------

    # res3c_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3c_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3b'])

    # res3c_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3c_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3c_branch2a'])

    # res3c_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3c_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3c_branch2a'])

    # res3c_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3c_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3c_branch2b'])

    # res3c_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3c_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3c_branch2b'])

    # res3c_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3c_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3c_branch2c'])

    # res3c residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3c',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3c_branch2c', 'res3b'])

    # ------------------- Residual Layer 3D -----------------------

    # res3d_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3d_branch2a',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res3c'])

    # res3d_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3d_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3d_branch2a'])

    # res3d_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3d_branch2b',
                         layer=dict(type='convolution', nFilters=128, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3d_branch2a'])

    # res3d_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3d_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res3d_branch2b'])

    # res3d_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3d_branch2c',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn3d_branch2b'])

    # res3d_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='bn3d_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res3d_branch2c'])

    # res3d residual layer: 512 channels, output = 28 x 28
    s.deepLearn.addLayer(model=model_table_opts, name='res3d',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn3d_branch2c', 'res3c'])

    # ------------- Layer 4A --------------------

    # res4a_branch1 layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch1',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res3d'])

    # res4a_branch1 batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4a_branch1'])

    # res4a_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4a_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res3d'])

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

    # ------------------- Residual Layer 4B -----------------------

    # res4b_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4a'])

    # res4b_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b_branch2a'])

    # res4b_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b_branch2a'])

    # res4b_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4b_branch2b'])

    # res4b_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4b_branch2b'])

    # res4b_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4b_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4b_branch2c'])

    # res4b residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4b',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4b_branch2c', 'res4a'])

    # ------------------- Residual Layer 4C -----------------------

    # res4c_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4c_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4b'])

    # res4c_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4c_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4c_branch2a'])

    # res4c_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4c_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4c_branch2a'])

    # res4c_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4c_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4c_branch2b'])

    # res4c_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4c_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4c_branch2b'])

    # res4c_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4c_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4c_branch2c'])

    # res4c residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4c',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4c_branch2c', 'res4b'])

    # ------------------- Residual Layer 4D -----------------------

    # res4d_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4d_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4c'])

    # res4d_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4d_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4d_branch2a'])

    # res4d_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4d_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4d_branch2a'])

    # res4d_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4d_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4d_branch2b'])

    # res4d_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4d_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4d_branch2b'])

    # res4d_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4d_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4d_branch2c'])

    # res4d residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4d',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4d_branch2c', 'res4c'])

    # ------------------- Residual Layer 4E ----------------------- */

    # res4e_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4e_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4d'])

    # res4e_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4e_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4e_branch2a'])

    # res4e_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4e_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4e_branch2a'])

    # res4e_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4e_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4e_branch2b'])

    # res4e_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4e_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4e_branch2b'])

    # res4e_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4e_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4e_branch2c'])

    # res4e residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4e',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4e_branch2c', 'res4d'])

    # ------------------- Residual Layer 4F -----------------------

    # res4f_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4f_branch2a',
                         layer=dict(type='convolution', nFilters=256, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['res4e'])

    # res4f_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4f_branch2a',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4f_branch2a'])

    # res4f_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4f_branch2b',
                         layer=dict(type='convolution', nFilters=256, width=3, height=3,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4f_branch2a'])

    # res4f_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4f_branch2b',
                         layer=dict(type='batchnorm', act='relu'),
                         srcLayers=['res4f_branch2b'])

    # res4f_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4f_branch2c',
                         layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                                    stride=1, includebias=False, act='identity'),
                         srcLayers=['bn4f_branch2b'])

    # res4f_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='bn4f_branch2c',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res4f_branch2c'])

    # res4f residual layer: 1024 channels, output = 14 x 14
    s.deepLearn.addLayer(model=model_table_opts, name='res4f',
                         layer=dict(type='residual', act='relu'),
                         srcLayers=['bn4f_branch2c', 'res4e'])

    # ------------- Layer 5A -------------------- */

    # res5a_branch1 layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch1',
                         layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res4f'])

    # res5a_branch1 batch norm layer: 2048 channels, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='bn5a_branch1',
                         layer=dict(type='batchnorm', act='identity'),
                         srcLayers=['res5a_branch1'])

    # res5a_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.deepLearn.addLayer(model=model_table_opts, name='res5a_branch2a',
                         layer=dict(type='convolution', nFilters=512, width=1, height=1,
                                    stride=2, includebias=False, act='identity'),
                         srcLayers=['res4f'])

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
