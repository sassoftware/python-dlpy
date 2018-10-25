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


def InceptionV3_Model(s, model_table='INCEPTIONV3', n_channels=3, width=299,
                      height=299, random_crop=None, offsets=None):
    '''
    InceptionV3 model definition

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
        Default: 299
    height : int, optional
        Specifies the height of the input layer.
        Default: 299
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters.deepLearn. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'unique'
        Default: 'unique'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.deepLearn.
        Default: (1, 1, 1)

    '''

    model_table_opts = input_table_check(model_table)

    if random_crop is None:
        random_crop = 'none'
    elif random_crop.lower() not in ['none', 'unique']:
        raise ValueError('random_crop can only be "none" or "unique"')

    scale = 1/127.5

    if offsets is None:
        offsets = [1, 1, 1]

    # instantiate model
    build_model_table_opts = dict(model_table_opts)
    build_model_table_opts['replace'] = True
    s.deepLearn.buildModel(model=build_model_table_opts, type='CNN')

    # input layer
    s.deepLearn.addLayer(model=model_table_opts, name='input_1',
                         layer=dict(type='input', nchannels=n_channels, width=width,
                                    height=height, randomcrop=random_crop, offsets=offsets,
                                    scale=scale))

    # 299 x 299 x 3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_1',
                         layer=dict(type='convolution', nFilters=32, width=3, height=3,
                                    stride=2, act='identity', includebias=False, padding=0),
                         srcLayers=['input_1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_1',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_1'])

    # 149 x 149 x 32
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_2',
                         layer=dict(type='convolution', nFilters=32, width=3, height=3,
                                    stride=1, act='identity', includebias=False, padding=0),
                         srcLayers=['batch_normalization_1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_2',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_2'])

    # 147 x 147 x 32
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_3',
                         layer=dict(type='convolution', nFilters=64, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_3',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_3'])

    # 147 x 147 x 64
    s.deepLearn.addLayer(model=model_table_opts, name='max_pooling2d_1',
                         layer=dict(type='pooling', width=3, height=3, stride=2, pool='max',
                                    padding=0),
                         srcLayers=['batch_normalization_3'])

    # 73 x 73 x 64
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_4',
                         layer=dict(type='convolution', nFilters=80, width=1, height=1,
                                    stride=1, act='identity', includebias=False, padding=0),
                         srcLayers=['max_pooling2d_1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_4',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_4'])

    # 73 x 73 x 80
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_5',
                         layer=dict(type='convolution', nFilters=192, width=3, height=3,
                                    stride=1, act='identity', includebias=False, padding=0),
                         srcLayers=['batch_normalization_4'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_5',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_5'])

    # 71 x 71 x 192
    s.deepLearn.addLayer(model=model_table_opts, name='max_pooling2d_2',
                         layer=dict(type='pooling', width=3, height=3, stride=2, pool='max',
                                    padding=0),
                         srcLayers=['batch_normalization_5'])

    # mixed 0: output 35 x 35 x 256

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_6',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['max_pooling2d_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_6',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_6'])

    # branch5x5
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_7',
                         layer=dict(type='convolution', nFilters=48, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['max_pooling2d_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_7',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_7'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_8',
                         layer=dict(type='convolution', nFilters=64, width=5, height=5,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_7'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_8',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_8'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_9',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['max_pooling2d_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_9',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_9'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_10',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_9'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_10',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_10'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_11',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_10'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_11',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_11'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_1',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['max_pooling2d_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_12',
                         layer=dict(type='convolution', nFilters=32, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_12',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_12'])

    # mixed0 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed0',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_6', 'batch_normalization_8',
                                    'batch_normalization_11', 'batch_normalization_12'])

    # mixed 1: output 35 x 35 x 288

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_13',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed0'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_13',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_13'])

    # branch5x5
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_14',
                         layer=dict(type='convolution', nFilters=48, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed0'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_14',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_14'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_15',
                         layer=dict(type='convolution', nFilters=64, width=5, height=5,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_14'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_15',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_15'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_16',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed0'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_16',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_16'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_17',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_16'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_17',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_17'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_18',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_17'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_18',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_18'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_2',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed0'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_19',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_19',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_19'])

    # mixed1 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed1',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_13', 'batch_normalization_15',
                                    'batch_normalization_18', 'batch_normalization_19'])

    # mixed 2: output 35 x 35 x 288

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_20',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_20',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_20'])

    # branch5x5
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_21',
                         layer=dict(type='convolution', nFilters=48, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_21',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_21'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_22',
                         layer=dict(type='convolution', nFilters=64, width=5, height=5,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_21'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_22',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_22'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_23',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed1'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_23',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_23'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_24',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_23'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_24',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_24'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_25',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_24'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_25',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_25'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_3',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed1'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_26',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_3'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_26',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_26'])

    # mixed2 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed2',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_20', 'batch_normalization_22',
                                    'batch_normalization_25', 'batch_normalization_26'])

    # mixed 3: output 17 x 17 x 768

    # branch3x3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_27',
                         layer=dict(type='convolution', nFilters=384, width=3, height=3,
                                    stride=2, act='identity', includebias=False, padding=0),
                         srcLayers=['mixed2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_27',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_27'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_28',
                         layer=dict(type='convolution', nFilters=64, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed2'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_28',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_28'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_29',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_28'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_29',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_29'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_30',
                         layer=dict(type='convolution', nFilters=96, width=3, height=3,
                                    stride=2, act='identity', includebias=False, padding=0),
                         srcLayers=['batch_normalization_29'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_30',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_30'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='max_pooling2d_3',
                         layer=dict(type='pooling', width=3, height=3, stride=2, pool='max',
                                    padding=0),
                         srcLayers=['mixed2'])

    # mixed3 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed3',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_27', 'batch_normalization_30',
                                    'max_pooling2d_3'])

    # mixed 4: output 17 x 17 x 768

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_31',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed3'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_31',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_31'])

    # branch7x7
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_32',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed3'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_32',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_32'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_33',
                         layer=dict(type='convolution', nFilters=128, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_32'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_33',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_33'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_34',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_33'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_34',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_34'])

    # branch7x7dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_35',
                         layer=dict(type='convolution', nFilters=128, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed3'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_35',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_35'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_36',
                         layer=dict(type='convolution', nFilters=128, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_35'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_36',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_36'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_37',
                         layer=dict(type='convolution', nFilters=128, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_36'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_37',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_37'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_38',
                         layer=dict(type='convolution', nFilters=128, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_37'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_38',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_38'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_39',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_38'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_39',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_39'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_4',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed3'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_40',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_4'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_40',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_40'])

    # mixed4 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed4',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_31', 'batch_normalization_34',
                                    'batch_normalization_39', 'batch_normalization_40'])

    # mixed 5

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_41',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed4'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_41',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_41'])

    # branch7x7
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_42',
                         layer=dict(type='convolution', nFilters=160, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed4'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_42',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_42'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_43',
                         layer=dict(type='convolution', nFilters=160, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_42'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_43',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_43'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_44',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_43'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_44',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_44'])

    # branch7x7dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_45',
                         layer=dict(type='convolution', nFilters=160, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed4'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_45',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_45'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_46',
                         layer=dict(type='convolution', nFilters=160, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_45'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_46',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_46'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_47',
                         layer=dict(type='convolution', nFilters=160, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_46'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_47',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_47'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_48',
                         layer=dict(type='convolution', nFilters=160, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_47'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_48',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_48'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_49',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_48'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_49',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_49'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_5',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed4'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_50',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_5'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_50',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_50'])

    # mixed5 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed5',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_41', 'batch_normalization_44',
                                    'batch_normalization_49', 'batch_normalization_50'])

    # mixed6: output 17 x 17 x 768

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_51',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed5'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_51',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_51'])

    # branch7x7
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_52',
                         layer=dict(type='convolution', nFilters=160, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed5'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_52',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_52'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_53',
                         layer=dict(type='convolution', nFilters=160, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_52'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_53',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_53'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_54',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_53'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_54',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_54'])

    # branch7x7dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_55',
                         layer=dict(type='convolution', nFilters=160, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed5'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_55',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_55'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_56',
                         layer=dict(type='convolution', nFilters=160, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_55'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_56',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_56'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_57',
                         layer=dict(type='convolution', nFilters=160, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_56'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_57',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_57'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_58',
                         layer=dict(type='convolution', nFilters=160, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_57'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_58',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_58'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_59',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_58'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_59',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_59'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_6',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed5'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_60',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_6'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_60',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_60'])

    # mixed6 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed6',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_51', 'batch_normalization_54',
                                    'batch_normalization_59', 'batch_normalization_60'])

    # mixed 7: output 17 x 17 x 768

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_61',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed6'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_61',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_61'])

    # branch7x7
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_62',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed6'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_62',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_62'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_63',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_62'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_63',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_63'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_64',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_63'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_64',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_64'])

    # branch7x7dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_65',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed6'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_65',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_65'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_66',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_65'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_66',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_66'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_67',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_66'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_67',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_67'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_68',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_67'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_68',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_68'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_69',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_68'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_69',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_69'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_7',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed6'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_70',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_7'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_70',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_70'])

    # mixed7 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed7',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_61', 'batch_normalization_64',
                                    'batch_normalization_69', 'batch_normalization_70'])

    # mixed 8: output 8 x 8 x 1280

    # branch3x3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_71',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed7'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_71',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_71'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_72',
                         layer=dict(type='convolution', nFilters=320, width=3, height=3,
                                    stride=2, act='identity', includebias=False, padding=0),
                         srcLayers=['batch_normalization_71'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_72',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_72'])

    # branch7x7x3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_73',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed7'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_73',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_73'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_74',
                         layer=dict(type='convolution', nFilters=192, width=7, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_73'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_74',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_74'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_75',
                         layer=dict(type='convolution', nFilters=192, width=1, height=7,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_74'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_75',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_75'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_76',
                         layer=dict(type='convolution', nFilters=192, width=3, height=3,
                                    stride=2, act='identity', includebias=False, padding=0),
                         srcLayers=['batch_normalization_75'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_76',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_76'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='max_pooling2d_4',
                         layer=dict(type='pooling', width=3, height=3, stride=2, pool='max',
                                    padding=0),
                         srcLayers=['mixed7'])

    # mixed8 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed8',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_72', 'batch_normalization_76',
                                    'max_pooling2d_4'])

    # mixed 9

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_77',
                         layer=dict(type='convolution', nFilters=320, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed8'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_77',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_77'])

    # branch3x3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_78',
                         layer=dict(type='convolution', nFilters=384, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed8'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_78',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_78'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_79',
                         layer=dict(type='convolution', nFilters=384, width=3, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_78'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_79',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_79'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_80',
                         layer=dict(type='convolution', nFilters=384, width=1, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_78'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_80',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_80'])

    s.deepLearn.addLayer(model=model_table_opts, name='mixed9_0',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_79', 'batch_normalization_80'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_81',
                         layer=dict(type='convolution', nFilters=448, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed8'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_81',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_81'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_82',
                         layer=dict(type='convolution', nFilters=384, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_81'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_82',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_82'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_83',
                         layer=dict(type='convolution', nFilters=384, width=3, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_82'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_83',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_83'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_84',
                         layer=dict(type='convolution', nFilters=384, width=1, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_82'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_84',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_84'])

    s.deepLearn.addLayer(model=model_table_opts, name='concatenate_1',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_83', 'batch_normalization_84'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_8',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed8'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_85',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_8'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_85',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_85'])

    # mixed9 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed9',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_77', 'mixed9_0',
                                    'concatenate_1', 'batch_normalization_85'])

    # mixed 10:  output 8 x 8 x 2048

    # branch1x1
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_86',
                         layer=dict(type='convolution', nFilters=320, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed9'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_86',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_86'])

    # branch3x3
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_87',
                         layer=dict(type='convolution', nFilters=384, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed9'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_87',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_87'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_88',
                         layer=dict(type='convolution', nFilters=384, width=3, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_87'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_88',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_88'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_89',
                         layer=dict(type='convolution', nFilters=384, width=1, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_87'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_89',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_89'])

    s.deepLearn.addLayer(model=model_table_opts, name='mixed9_1',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_88', 'batch_normalization_89'])

    # branch3x3dbl
    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_90',
                         layer=dict(type='convolution', nFilters=448, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['mixed9'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_90',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_90'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_91',
                         layer=dict(type='convolution', nFilters=384, width=3, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_90'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_91',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_91'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_92',
                         layer=dict(type='convolution', nFilters=384, width=3, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_91'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_92',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_92'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_93',
                         layer=dict(type='convolution', nFilters=384, width=1, height=3,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['batch_normalization_91'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_93',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_93'])

    s.deepLearn.addLayer(model=model_table_opts, name='concatenate_2',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_92', 'batch_normalization_93'])

    # branch_pool
    s.deepLearn.addLayer(model=model_table_opts, name='average_pooling2d_9',
                         layer=dict(type='pooling', width=3, height=3, stride=1, pool='average'),
                         srcLayers=['mixed9'])

    s.deepLearn.addLayer(model=model_table_opts, name='conv2d_94',
                         layer=dict(type='convolution', nFilters=192, width=1, height=1,
                                    stride=1, act='identity', includebias=False),
                         srcLayers=['average_pooling2d_9'])

    s.deepLearn.addLayer(model=model_table_opts, name='batch_normalization_94',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2d_94'])

    # mixed10 concat
    s.deepLearn.addLayer(model=model_table_opts, name='mixed10',
                         layer=dict(type='concat', act='identity'),
                         srcLayers=['batch_normalization_86', 'mixed9_1',
                                    'concatenate_2', 'batch_normalization_94'])

    # calculate dimensions for global average pooling
    w = (width - 75) // 32 + 1
    h = (height - 75) // 32 + 1

    # global average pooling
    s.deepLearn.addLayer(model=model_table_opts, name='avg_pool',
                         layer=dict(type='pooling', width=w, height=h, stride=1, pool='average',
                                    padding=0),
                         srcLayers=['mixed10'])

    # output layer
    s.deepLearn.addLayer(model=model_table_opts, name='predictions',
                         layer=dict(type='output', n=1000, act='softmax'),
                         srcLayers=['avg_pool'])

    return s.CASTable(**model_table_opts)
