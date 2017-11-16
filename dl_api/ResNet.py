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
Residual Network object for deep learning.
'''

from .layers import *


class Block:
    '''
    This is a module of a block of layers that have special connectivity instead of sequential.
    '''

    def __init__(self):
        self.layers = []
        self.connectivity = {}


class ResBlock:
    '''
    Residual block for Residual Network.
    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None):
        '''

        :param kernel_size: kernel_size of the convolution filters.
        :param n_filters: list of numbers of filter in each convolution layers.
        :param strides: list of stride in each convolution layers.
        :return: ResBlock object
        '''

        if strides is None:
            strides = [1] * len(n_filters)
        elif len(strides) == 1:
            strides = [strides].append([1] * (len(n_filters) - 1))

        if len(strides) != len(n_filters):
            raise ValueError('The length of strides must be equal '
                             'to the length of n_filters.')
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes]
        if len(kernel_sizes) == 1:
            kernel_sizes = [kernel_sizes] * len(n_filters)
        elif len(kernel_sizes) != len(n_filters):
            raise ValueError('The length of kernel_sizes must be equal '
                             'to the length of n_filters.')
        self.config = dict()
        self.config['type'] = 'block'
        self.layers = []

        for n_filter, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
            self.layers.append(Conv2d(n_filters=n_filter,
                                      width=kernel_size,
                                      stride=stride))

        self.layers.append(Res(act='identity'))

    def compile(self, src_layer, block_num):
        options = []
        conv_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.config['type'].lower() in ('convo', 'convolution'):
                layer.name = 'R{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]

            elif layer.config['type'].lower() == 'residual':
                layer.name = 'Res{}'.format(block_num)
                layer.src_layers = [input_layer, src_layer]
            input_layer = layer
            options.append(layer.to_model_params())
        return options


class ResBlockBN:
    '''
    Residual block for Residual Network.
    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None, batch_norm_first=True):

        if strides is None:
            strides = [1] * len(n_filters)

        if isinstance(strides, int):
            strides = [strides] + [1] * (len(n_filters) - 1)

        if len(strides) != len(n_filters):
            raise ValueError('The length of strides must be equal '
                             'to the length of n_filters.')
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(n_filters)

        if len(kernel_sizes) != len(n_filters):
            raise ValueError('The length of kernel_sizes must be equal '
                             'to the length of n_filters.')
        self.config = dict()
        self.config['type'] = 'block'
        self.layers = []
        if batch_norm_first:
            for n_filter, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
                self.layers.append(BN(act='relu'))
                self.layers.append(
                    Conv2d(n_filters=n_filter,
                           width=kernel_size,
                           act='identity',
                           stride=stride,
                           includeBias=False))
        else:
            for n_filter, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
                self.layers.append(
                    Conv2d(n_filters=n_filter,
                           width=kernel_size,
                           act='identity',
                           stride=stride,
                           includeBias=False))
                self.layers.append(BN(act='relu'))

        self.layers.append(Res(act='identity'))

    def compile(self, src_layer, block_num):
        options = []
        conv_num = 1
        bn_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.config['type'].lower() in ('convo', 'convolution'):
                layer.name = 'R{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]
            elif layer.config['type'].lower() == 'batchnorm':
                layer.name = 'R{}B{}'.format(block_num, bn_num)
                bn_num += 1
                layer.src_layers = [input_layer]
            elif layer.config['type'].lower() == 'residual':
                layer.name = 'Res{}'.format(block_num)
                layer.src_layers = [input_layer, src_layer]
            input_layer = layer
            options.append(layer.to_model_params())
        return options


class ResBlock_Caffe:
    '''
    Residual block for Residual Network.
    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None, batch_norm_first=False,
                 conv_short_cut=False):

        if strides is None:
            strides = [1] * len(n_filters)

        if isinstance(strides, int):
            strides = [strides] + [1] * (len(n_filters) - 1)

        if len(strides) != len(n_filters):
            raise ValueError('The length of strides must be equal '
                             'to the length of n_filters.')
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(n_filters)

        if len(kernel_sizes) != len(n_filters):
            raise ValueError('The length of kernel_sizes must be equal '
                             'to the length of n_filters.')
        self.config = dict()
        self.config['type'] = 'block'
        self.layers = []
        self.conv_short_cut = conv_short_cut
        if batch_norm_first:
            if conv_short_cut:
                self.layers.append(BN(act='relu'))
                self.layers.append(
                    Conv2d(n_filters=n_filters[-1],
                           width=1,
                           act='identity',
                           stride=strides[0],
                           includeBias=False))
            self.layers.append(Res(act='identity'))

            for n_filter, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
                self.layers.append(BN(act='relu'))
                self.layers.append(
                    Conv2d(n_filters=n_filter,
                           width=kernel_size,
                           act='identity',
                           stride=stride,
                           includeBias=False))
        else:
            if conv_short_cut:
                self.layers.append(
                    Conv2d(n_filters=n_filters[-1],
                           width=1,
                           act='identity',
                           stride=strides[0],
                           includeBias=False))
                self.layers.append(BN(act='identity'))

            for n_filter, kernel_size, stride, act in zip(n_filters, kernel_sizes, strides,
                                                          ['relu', 'relu', 'identity']):
                self.layers.append(
                    Conv2d(n_filters=n_filter,
                           width=kernel_size,
                           act='identity',
                           stride=stride,
                           includeBias=False))
                self.layers.append(BN(act=act))

            self.layers.append(Res(act='relu'))

    def compile(self, src_layer, block_num):
        options = []
        conv_num = 1
        bn_num = 1
        input_layer = src_layer
        if self.conv_short_cut:
            for layer in self.layers[:2]:
                if layer.config['type'].lower() in ('convo', 'convolution'):
                    layer.name = 'R{}C{}'.format(block_num, 0)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.config['type'].lower() == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, 0)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                input_layer = layer
                options.append(layer.to_model_params())
            short_cut_layer = layer
            input_layer = src_layer
            for layer in self.layers[2:]:
                if layer.config['type'].lower() in ('convo', 'convolution'):
                    layer.name = 'R{}C{}'.format(block_num, conv_num)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.config['type'].lower() == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, bn_num)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                elif layer.config['type'].lower() == 'residual':
                    layer.name = 'Res{}'.format(block_num)
                    layer.src_layers = [input_layer, short_cut_layer]
                input_layer = layer
                options.append(layer.to_model_params())
        else:
            for layer in self.layers:
                if layer.config['type'].lower() in ('convo', 'convolution'):
                    layer.name = 'R{}C{}'.format(block_num, conv_num)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.config['type'].lower() == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, bn_num)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                elif layer.config['type'].lower() == 'residual':
                    layer.name = 'Res{}'.format(block_num)
                    layer.src_layers = [input_layer, src_layer]
                input_layer = layer
                options.append(layer.to_model_params())
        return options
