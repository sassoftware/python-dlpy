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
from .utils import prod_without_none


class Layer:
    def __init__(self, name=None, config=None, src_layers=None):
        self.name = name
        self.config = config
        self.src_layers = src_layers

        self.output_size = None
        self.kernel_size = None
        self.num_weights = None
        self.num_bias = None

    def summary(self):
        # Note: this will be moved to complie.
        if self.config['type'].lower() == 'input':
            self.output_size = (self.config['width'], self.config['height'], self.config['nchannels'])
            self.kernel_size = None
            self.num_weights = 0
            self.num_bias = 0
        elif self.config['type'].lower() in ('convo', 'convolution'):
            self.output_size = (self.src_layers.output_size[0] // self.config['stride'],
                                self.src_layers.output_size[1] // self.config['stride'],
                                self.config['nfilters'])
            self.kernel_size = (self.config['width'], self.config['height'])
            self.num_weights = self.config['width'] * self.config['height'] * self.config['nfilters'] * \
                               self.src_layers.output_size[2]
            self.num_bias = self.config['nfilters']
        elif self.config['type'].lower() in ('pool', 'pooling'):
            self.output_size = (self.src_layers.output_size[0] // self.config['stride'],
                                self.src_layers.output_size[1] // self.config['stride'],
                                self.src_layers.output_size[2])
            self.kernel_size = (self.config['width'], self.config['height'])
            self.num_weights = 0
            self.num_bias = 0
        elif self.config['type'].lower() in ('fc', 'fullconnect'):
            if isinstance(self.src_layers.output_size, int):
                num_features = self.src_layers.output_size
            else:
                num_features = prod_without_none(self.src_layers.output_size)

            self.output_size = (self.config['n'])
            self.kernel_size = (num_features, self.config['n'])
            self.num_weights = num_features * self.config['n']
            self.num_bias = self.config['n']
        elif self.config['type'].lower() == 'output':
            if isinstance(self.src_layers.output_size, int):
                num_features = self.src_layers.output_size
            else:
                num_features = prod_without_none(self.src_layers.output_size)

            if 'n' in self.config.keys():
                self.output_size = (self.config['n'])
                self.kernel_size = (num_features, self.config['n'])
                self.num_weights = num_features * self.config['n']
                self.num_bias = self.config['n']
            else:
                self.kernel_size = None
                self.num_weights = None
                self.num_bias = None
                self.output_size = None

        name = '{}({})'.format(self.name, self.config['type'])
        col1 = '| {:<14}'.format('{}'.format(name))
        col2 = '|{:^15}'.format('{}'.format(self.kernel_size))
        if 'stride' not in self.config.keys():
            col3 = '|{:^8}'.format('None')
        else:
            col3 = '|{:^8}'.format('{}'.format(self.config['stride']))
        if 'act' not in self.config.keys():
            col4 = '|{:^12}'.format('None')
        else:
            col4 = '|{:^12}'.format('{}'.format(self.config['act']))
        col5 = '|{:^17}'.format('{}'.format(self.output_size))
        num_paras = '{} / {}'.format(self.num_weights, self.num_bias)
        col6 = '|{:^22}|\n'.format(num_paras)

        return col1 + col2 + col3 + col4 + col5 + col6


class InputLayer(Layer):
    def __init__(self, n_channels=3, width=224, height=224, scale=1, dropout=0, offsets='NONE',
                 name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'input'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#F0FF00'


class Conv2d(Layer):
    def __init__(self, n_filters, width=None, height=None, stride=1, act="relu", dropout=0,
                 name=None, src_layers=None, **kwargs):
        if (width is None) and (height is None):
            width = 3
        if width is None:
            width = height
        if height is None:
            height = width
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'convo'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#6CFF00'


class Pooling(Layer):
    def __init__(self, width=None, height=None, stride=None, pool='max', dropout=0,
                 name=None, src_layers=None, **kwargs):
        if (width is None) and (height is None):
            width = 2
        if width is None:
            width = height
        if height is None:
            height = width
        if stride is None:
            stride = width
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'pool'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FF9700'


class Dense(Layer):
    def __init__(self, n, act='relu', dropout=0,
                 name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'fc'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#00ECFF'


class Recurrent(Layer):
    def __init__(self, n, act='AUTO', rnn_type='RNN', output_type='ENCODING', reversed_=False,
                 name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'recurrent'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FFA4A4'


class OutputLayer(Layer):
    def __init__(self, n=None, act='softmax', name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'output'
        if config['n'] is None:
            del config['n']
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#C8C8C8'


def _unpack_config(config):
    kwargs = config['kwargs']
    del config['self'], config['name'], config['src_layers'], config['kwargs']
    out = {}
    out.update(config)
    out.update(kwargs)
    for key in out:
        if '_' in key:
            new_key = key.replace('_', '')
            out[new_key] = out[key]
            out.pop(key)
    return out
