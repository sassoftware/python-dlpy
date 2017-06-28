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

from .model import Model


def VGG16(sess, model_name, pre_train_weight=False, include_top=False,
          nchannels=3, width=224, height=224, scale=1,
          randomflip='HV', randomcrop='unique', offsets=(85, 111, 139)):
    '''
    Function to generate a deep learning model with VGG16 architecture.

    Parameters:

    ----------
    sess :
        Specifies the session of the CAS connection.
    model_name : string
        Specifies the name of CAS table to store the model.
    pre_train_weight : boolean, optional.
        Specifies whether to use the pre-trained weights from ImageNet data set.
        Default : False.
    include_top : boolean, optional.
        Specifies whether to include pre-trained weights of the top layers, i.e. the FC layers.
        Default : False.
    nchannels : double, optional.
        Specifies the number of the channels of the input layer.
        Default : 3.
    width : double, optional.
        Specifies the width of the input layer.
        Default : 224.
    height : double, optional.
        Specifies the height of the input layer.
        Default : 224.
    scale : double, optional.
        Specifies a scaling factor to apply to each image..
        Default : 1.
    randomFlip : string, "H" | "HV" | "NONE" | "V"
        Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
        is subject to flipping.
        Default	: "HV"
    randomcrop : string, "NONE" or "UNIQUE"
        Specifies how to crop the data in the input layer when image data is used. Images are cropped to the values that
         are specified in the width and height parameters. Only the images with one or both dimensions that are larger
         than those sizes are cropped.
        Default	: "UNIQUE"
    offsets=(double-1 <, double-2, ...>), optional
        Specifies an offset for each channel in the input data. The final input data is set after applying scaling and
        subtracting the specified offsets.
    Default : (85, 111, 139)

    Returns
    -------
    A model object using VGG16 architecture.

    '''

    _VGG16_(sess, model_name, nchannels=nchannels, width=width, height=height,
            scale=scale, randomflip=randomflip, randomcrop=randomcrop,
            offsets=offsets)

    if pre_train_weight:
        sess.loadtable(casout={'replace': True, 'name': '_imagenet_vgg16_weights'},
                       caslib='CASTestTmp', path='vgg/vgg16_weights_table.sashdat')
        if include_top:
            model = Model(sess, model_name, '_imagenet_vgg16_weights')
        else:
            model = Model(sess, model_name, model_weights=dict(name='_imagenet_vgg16_weights',
                                                               where='_layerid_<18'))
    else:
        model = Model(sess, model_name)
    return model


def _VGG16_(sess, model_name, nchannels=3, width=224, height=224, scale=1,
            randomflip='HV', randomcrop='unique', offsets=(85, 111, 139)):
    if not sess.queryactionset('deepLearn')['deepLearn']:
        sess.loadactionset('deepLearn')

    sess.buildmodel(model=dict(name=model_name, replace=True), type='CNN')
    sess.addLayer(model=model_name, name='data',
                  layer=dict(type='input', nchannels=nchannels, width=width, height=height,
                             scale=scale, randomflip=randomflip, randomcrop=randomcrop,
                             offsets=offsets)
                  )

    # conv1
    sess.addLayer(model=model_name, name='conv1_1',
                  layer=dict(type='convolution', nFilters=64, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['data'])
    sess.addLayer(model=model_name, name='conv1_2',
                  layer=dict(type='convolution', nFilters=64, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv1_1'])
    sess.addLayer(model=model_name, name='pool1',
                  layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                  srcLayers=['conv1_2'])

    # conv2
    sess.addLayer(model=model_name, name='conv2_1',
                  layer=dict(type='convolution', nFilters=128, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['pool1'])
    sess.addLayer(model=model_name, name='conv2_2',
                  layer=dict(type='convolution', nFilters=128, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv2_1'])
    sess.addLayer(model=model_name, name='pool2',
                  layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                  srcLayers=['conv2_2'])

    # conv3
    sess.addLayer(model=model_name, name='conv3_1',
                  layer=dict(type='convolution', nFilters=256, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['pool2'])
    sess.addLayer(model=model_name, name='conv3_2',
                  layer=dict(type='convolution', nFilters=256, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv3_1'])
    sess.addLayer(model=model_name, name='conv3_3',
                  layer=dict(type='convolution', nFilters=256, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv3_2'])
    sess.addLayer(model=model_name, name='pool3',
                  layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                  srcLayers=['conv3_3'])

    # conv4
    sess.addLayer(model=model_name, name='conv4_1',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['pool3'])
    sess.addLayer(model=model_name, name='conv4_2',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv4_1'])
    sess.addLayer(model=model_name, name='conv4_3',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv4_2'])
    sess.addLayer(model=model_name, name='pool4',
                  layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                  srcLayers=['conv4_3'])

    # conv5
    sess.addLayer(model=model_name, name='conv5_1',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['pool4'])
    sess.addLayer(model=model_name, name='conv5_2',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv5_1'])
    sess.addLayer(model=model_name, name='conv5_3',
                  layer=dict(type='convolution', nFilters=512, width=3, height=3, stride=1,
                             init="XAVIER", std=1e-1, truncfact=2, act="relu"),
                  srcLayers=['conv5_2'])
    sess.addLayer(model=model_name, name='pool5',
                  layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                  srcLayers=['conv5_3'])

    sess.addLayer(model=model_name, name='fc1',
                  layer=dict(type='fullconnect', n=4096, act='relu', std=1e-1, truncfact=2),
                  srcLayers=['pool5'])
    sess.addLayer(model=model_name, name='fc2',
                  layer=dict(type='fullconnect', n=4096, act='relu', std=1e-1, truncfact=2),
                  srcLayers=['fc1'])
    sess.addLayer(model=model_name, name='outlayer',
                  layer=dict(type='output', act='softmax'),
                  srcLayers=['fc2'])

    return sess.CASTable(model_name)
