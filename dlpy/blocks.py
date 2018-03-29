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

''' Residual Network object for deep learning '''

from .layers import (Conv2d, BN, Res, Concat)


class Block(object):
    ''' Block of layers that have special connectivity instead of sequential '''

    def __init__(self):
        self.layers = []
        self.connectivity = {}


class ResBlock(object):
    '''
    Residual block for Residual Network

    Parameters
    ----------
    kernel_size : int, optional.
        Kernel size of the convolution filters.
    n_filters : iter-of-ints, optional.
        List of numbers of filter in each convolution layers.
    strides : iter-of-ints, optional.
        List of stride in each convolution layers.

    Returns
    -------
    :class:`ResBlock`

    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None):
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
        '''
        Convert the options into DLPy layer definition.

        Parameters
        ----------
        src_layer : Layer object.
            The source layer for the whole block.
        block_num : int.
            The label of the block. (used to name the layers).

        Returns
        -------
        dict
            A dictionary of keyword-arguments.

        '''
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


class ResBlockBN(object):
    '''
    Residual block for Residual Network with batch normalization.

    Parameters
    ----------
    kernel_sizes : iter-of-ints, optional.
        Kernel size of the convolution filters.
    n_filters : iter-of-ints, optional.
        List of numbers of filter in each convolution layers.
    strides : iter-of-ints, optional.
        List of stride in each convolution layers.
    batch_norm_first : bool, optional.
        Specify whether to add batch normal layer before conv layer.

    Returns
    -------
    :class:`ResBlockBN`

    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None,
                 batch_norm_first=True):
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
        '''
        Convert the options into DLPy layer definition.

        Parameters
        ----------
        src_layer : Layer object.
            The source layer for the whole block.
        block_num : int
            The label of the block. (used to name the layers)

        Returns
        -------
        dict
            A dictionary of keyword-arguments

        '''
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


class ResBlock_Caffe(object):
    '''
    Residual block for Residual Network with batch normalization.

    '''

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None,
                 batch_norm_first=False, conv_short_cut=False):
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

            for n_filter, kernel_size, stride, act in zip(n_filters,
                                                          kernel_sizes,
                                                          strides,
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
        '''
        Parameters
        ----------
        src_layer : Layer object.
            The source layer for the whole block.
        block_num : int
            The label of the block. (used to name the layers)

        Returns
        -------
        dict
            A dictionary of keyword-arguments

        '''
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


class DenseNetBlock(object):
    '''
    DenseNet block

    Parameters
    ----------
    n_cells : int
        Number of cells
    kernel_size : int
        Size of the kernel
    n_filter : int
        Number of filters
    stride : int
        Size of the stride

    Returns
    -------
    :class:`DenseNetBlock`

    '''

    def __init__(self, n_cells=4, kernel_size=3, n_filter=12, stride=1):
        self.config = dict()
        self.config['type'] = 'block'
        self.layers = []
        for _ in range(n_cells):
            self.layers.append(BN(act='relu'))
            self.layers.append(
                Conv2d(n_filters=n_filter,
                       width=kernel_size,
                       act='relu',
                       stride=stride,
                       includeBias=False))
            self.layers.append(Concat(act='identity'))

    def compile(self, src_layer, block_num):
        '''
        Convert the options into DLPy layer definition.

        Parameters
        ----------
        src_layer : Layer object.
            The source layer for the whole block.
        block_num : int
            The label of the block. (used to name the layers)

        Returns
        -------
        dict
            A dictionary of keyword-arguments

        '''
        options = []
        conv_num = 1
        bn_num = 1
        concat_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.config['type'].lower() in ('convo', 'convolution'):
                layer.name = 'D{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]
            elif layer.config['type'].lower() == 'batchnorm':
                layer.name = 'D{}B{}'.format(block_num, bn_num)
                bn_num += 1
                layer.src_layers = [input_layer]
            elif layer.config['type'].lower() == 'concat':
                layer.name = 'D{}Concat{}'.format(block_num, concat_num)
                concat_num += 1
                layer.src_layers = [input_layer, src_layer]
                src_layer = layer
            input_layer = layer
            options.append(layer.to_model_params())
        return options
