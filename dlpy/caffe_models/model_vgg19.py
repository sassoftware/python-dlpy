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


def VGG19_Model(s, model_name='VGG19', n_channels=3, width=224, height=224,
                random_crop=None, offsets=None, include_top=True):
    '''
    VGG19 model definition

    Parameters
    ----------
    s : CAS
        Specifies the CAS connection object
    model_name : string, optional
        Specifies the name of CAS table to store the model
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
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none' or 'UNIQUE'
        Default	: 'UNIQUE'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)

    Returns
    -------
    a CAS table defining the model is created.
    '''

    # TODO: Use underscore-delimited parameter names
    # quick error-checking and default setting
    if random_crop is None:
        random_crop = "NONE"
    elif random_crop.upper() not in ["NONE", "UNIQUE"]:
        raise ValueError('random_crop can only be "NONE" or "UNIQUE"')

    if (offsets == None):
        offsets = [103.939, 116.779, 123.68]

    # instantiate model
    s.buildModel(model=dict(name=model_name, replace=True), type='CNN')

    # input layer
    s.addLayer(model=model_name, name='data',
               layer=dict(type='input', nchannels=n_channels, width=width, height=height,
                          randomcrop=random_crop, offsets=offsets))

    # conv1_1 layer: 64*3*3
    s.addLayer(model=model_name, name='conv1_1',
               layer=dict(type='convolution', nFilters=64, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['data'])

    # conv1_2 layer: 64*3*3
    s.addLayer(model=model_name, name='conv1_2',
               layer=dict(type='convolution', nFilters=64, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv1_1'])

    # pool1 layer: 2*2
    s.addLayer(model=model_name, name='pool1',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv1_2'])

    # conv2_1 layer: 128*3*3
    s.addLayer(model=model_name, name='conv2_1',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['pool1'])

    # conv2_2 layer: 128*3*3
    s.addLayer(model=model_name, name='conv2_2',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv2_1'])

    # pool2 layer: 2*2
    s.addLayer(model=model_name, name='pool2',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv2_2'])

    # conv3_1 layer: 256*3*3
    s.addLayer(model=model_name, name='conv3_1',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['pool2'])

    # conv3_2 layer: 256*3*3
    s.addLayer(model=model_name, name='conv3_2',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv3_1'])

    # conv3_3 layer: 256*3*3
    s.addLayer(model=model_name, name='conv3_3',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv3_2'])

    # conv3_4 layer: 256*3*3
    s.addLayer(model=model_name, name='conv3_4',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv3_3'])

    # pool3 layer: 2*2
    s.addLayer(model=model_name, name='pool3',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv3_4'])

    # conv4_1 layer: 512*3*3
    s.addLayer(model=model_name, name='conv4_1',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['pool3'])

    # conv4_2 layer: 512*3*3
    s.addLayer(model=model_name, name='conv4_2',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv4_1'])

    # conv4_3 layer: 512*3*3
    s.addLayer(model=model_name, name='conv4_3',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv4_2'])

    # conv4_4 layer: 512*3*3 */
    s.addLayer(model=model_name, name='conv4_4',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv4_3'])

    # pool4 layer: 2*2
    s.addLayer(model=model_name, name='pool4',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv4_4'])

    # conv5_1 layer: 512*3*3
    s.addLayer(model=model_name, name='conv5_1',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['pool4'])

    # conv5_2 layer: 512*3*3
    s.addLayer(model=model_name, name='conv5_2',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv5_1'])

    # conv5_3 layer: 512*3*3
    s.addLayer(model=model_name, name='conv5_3',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv5_2'])

    # conv5_4 layer: 512*3*3
    s.addLayer(model=model_name, name='conv5_4',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, act='relu'),
               srcLayers=['conv5_3'])

    # pool5 layer: 2*2
    s.addLayer(model=model_name, name='pool5',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv5_4'])
    if include_top:
        # fc6 layer: 4096 neurons
        s.addLayer(model=model_name, name='fc6',
                   layer=dict(type='fullconnect', n=4096, act='relu', dropout=0.5),
                   srcLayers=['pool5'])

        # fc7 layer: 4096 neurons
        s.addLayer(model=model_name, name='fc7',
                   layer=dict(type='fullconnect', n=4096, act='relu', dropout=0.5),
                   srcLayers=['fc6'])
        # fc output layer: 1000 neurons
        s.addLayer(model=model_name, name='fc8',
                   layer=dict(type='output', n=1000, act='softmax'),
                   srcLayers=['fc7'])
