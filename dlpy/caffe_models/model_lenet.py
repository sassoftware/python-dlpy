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

''' LeNet with batch normalization model definition '''


def LeNet_Model(s, model_name='LeNet'):
    '''
    LeNet model definition (batch normalization version)

    Parameters
    ----------
    s : CAS
        Specifies the CAS connection object
    model_name : string, optional
        Specifies the name of CAS table to store the model
    include_top : boolean, optional
        Specifies whether to include the top layers of the model.

    Returns
    -------
    None
        A CAS table defining the model is created

    '''

    # instantiate model
    s.deepLearn.buildModel(model=dict(name=model_name, replace=True), type='CNN')

    # input layer
    s.deepLearn.addLayer(model=model_name, name='mnist',
                         layer=dict(type='input', n_channels=1, width=28, height=28,
                                    scale=0.00392156862745098039))

    # conv1: 5*5*20
    s.deepLearn.addLayer(model=model_name, name='conv1',
                         layer=dict(type='convolution', nFilters=20, width=5, height=5,
                                    stride=1, act='identity', noBias=True, init='xavier'),
                         srcLayers=['mnist'])

    # conv1 batch normalization
    s.deepLearn.addLayer(model=model_name, name='conv1_bn',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv1'])

    # pool1 2*2*2
    s.deepLearn.addLayer(model=model_name, name='pool1',
                         layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                         srcLayers=['conv1_bn'])

    # conv2: 5*5*50
    s.deepLearn.addLayer(model=model_name, name='conv2',
                         layer=dict(type='convolution', nFilters=50, width=5, height=5,
                                    stride=1, act='identity', noBias=True, init='xavier'),
                         srcLayers=['pool1'])

    # conv2 batch normalization
    s.deepLearn.addLayer(model=model_name, name='conv2_bn',
                         layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2'])

    # pool2 2*2*2
    s.deepLearn.addLayer(model=model_name, name='pool2',
                         layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
                         srcLayers=['conv2_bn'])

    # fully connected layer
    s.deepLearn.addLayer(model=model_name, name='ip1',
                         layer=dict(type='fullconnect', n=500, init='xavier', act='relu'),
                         srcLayers=['pool2'])
    # output layer
    s.deepLearn.addLayer(model=model_name, name='ip2',
                         layer=dict(type='output', n=10, init='xavier', act='softmax'),
                         srcLayers=['ip1'])
