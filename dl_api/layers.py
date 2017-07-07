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

'''
Define some common layers for the deep learning models
'''


def inputlayer(nchannels=3, width=224, height=224, scale=1, offsets=None, **kwargs):
    config = dict(type='input', nchannels=nchannels, width=width,
                  height=height, scale=scale, offsets=offsets, **kwargs)
    summary = dict(name=None, type='Input', size=[(width, height, nchannels)],
                   activation='Identity', n_weight='NA', n_bias='NA')
    return Layer(config=config, summary=summary)


def conv2d(nFilters, width=None, height=None, stride=1,
           init="XAVIER", act="relu", **kwargs):
    if (width is None) and (height is None):
        width = 3
    if width is None:
        width = height
    if height is None:
        height = width

    config = dict(type='convolution', nFilters=nFilters, width=width, height=height, stride=stride,
                  init=init, std=1e-1, truncfact=2, act=act, **kwargs)
    summary = dict(name=None, type='Conv2D', size=[(width, height, nFilters), stride],
                   activation=act, n_weight='NA', n_bias='NA')
    return Layer(config=config, summary=summary)


def pooling(width=None, height=None, stride=None, pool='max', **kwargs):
    if (width is None) and (height is None):
        width = 2
    if width is None:
        width = height
    if height is None:
        height = width
    if stride is None:
        stride = width
    config = dict(type='Pooling', width=width, height=height, stride=stride, pool=pool, **kwargs)
    summary = dict(name=None, type='pooling', size=[(width, height), stride],
                   activation=pool, n_weight='NA', n_bias='NA')
    return Layer(config=config, summary=summary)


def dense(n, act='relu', **kwargs):
    config = dict(type='fullconnect', n=n, act=act, **kwargs)
    summary = dict(name=None, type='Dense', size=[n],
                   activation='Identity', n_weight='NA', n_bias='NA')
    return Layer(config=config, summary=summary)


def outputlayer(act='softmax', **kwargs):
    config = dict(type='output', act=act, **kwargs)
    summary = dict(name=None, type='pooling', size=None,
                   activation=act, n_weight='NA', n_bias='NA')
    return Layer(config=config, summary=summary)


class Layer:
    def __init__(self, name=None, config=None, src_layers=None, summary=None):
        self.name = name
        self.config = config
        self.src_layers = src_layers
        if config is not None:
            self.type = config['type']
        else:
            self.type = None
        self.summary = summary
