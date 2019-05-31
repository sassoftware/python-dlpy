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

''' Block layers for deep learning '''

from .layers import (Conv2d, BN, Res, Concat, Recurrent, InputLayer)
from dlpy.utils import DLPyError


class ResBlock(object):
    '''
    Base class for the residual blocks.

    Parameters
    ----------
    kernel_sizes : list-of-ints, optional
        Specifies the size of the kernels. This assumes the kernels are square.
        Default: 3
    n_filters : list-of-ints, optional
        Specifies the number of filters.
        Default: (16, 16)
    strides : list-of-ints, optional
        Specifies the stride values for the filters
    batch_norm_first : bool, optional
        Set to True, if the batch normalization comes first
        Default: False
    conv_short_cut : bool, optional
        Set to True, if convolution layer has a short cut
        Default: False

    Returns
    -------
    :class:`ResBlock`

    '''
    type = 'block'
    type_desc = 'Residual block'
    can_be_last_layer = False
    number_of_instances = 0

    # To keep track of the instance counts this will be using later to assing a name if not set
    number_of_instances = 0

    def __init__(self, kernel_sizes=3, n_filters=(16,16), strides=None, batch_norm_first=False, conv_short_cut=False):
        self.count_instances()
        if strides is None:
            self.strides = [1] * len(n_filters)
        else:
            if isinstance(strides, int):
                self.strides = [strides] + [1] * (len(n_filters) - 1)
            elif isinstance(strides, list) or isinstance(strides, set) or isinstance(strides, tuple):
                if len(strides) == 1:
                    self.strides = [strides].append([1] * (len(n_filters) - 1))
                else:
                    self.strides = strides
            else:
                raise DLPyError('The strides parameter needs to be an integer or list of integers.')

            if len(self.strides) != len(n_filters):
                raise DLPyError('The length of strides must be equal to the length of n_filters.')

        self.kernel_sizes = kernel_sizes
        self.n_filters = n_filters

        if isinstance(self.kernel_sizes, int):
            self.kernel_sizes = [self.kernel_sizes]
        else:
            self.kernel_sizes = self.kernel_sizes

        if len(self.kernel_sizes) == 1:
            self.kernel_sizes = [self.kernel_sizes] * len(self.n_filters)
        elif len(self.kernel_sizes) != len(self.n_filters):
            raise DLPyError('The length of kernel_sizes must be equal to the length of n_filters.')

        self.batch_norm_first = batch_norm_first
        self.conv_short_cut = conv_short_cut
        self.layers = []
        self.add_layers()

    @classmethod
    def count_instances(cls):
        cls.number_of_instances += 1

    @classmethod
    def get_number_of_instances(cls):
        return cls.number_of_instances

    def add_layers(self):
        '''  Add the layers for the block '''
        for n_filter, kernel_size, stride in zip(self.n_filters, self.kernel_sizes, self.strides):
            self.layers.append(Conv2d(n_filters=n_filter, width=kernel_size, stride=stride))
        self.layers.append(Res(act='identity'))

    def compile(self, src_layer, block_num=None):
        '''
        Convert the block structure into DLPy layer definitions.

        Parameters
        ----------
        src_layer : Layer
            The source layer for the whole block.
        block_num : int, optional
            The label of the block. (used to name the layers)

        Returns
        -------
        list
            A list of keyword-arguments

        '''

        if block_num is None:
            block_num = self.get_number_of_instances()

        options = []
        conv_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.type == 'convo':
                layer.name = 'R{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]

            elif layer.type == 'residual':
                layer.name = 'Res{}'.format(block_num)
                layer.src_layers = [input_layer, src_layer]
            input_layer = layer
            options.append(layer.to_model_params())
        return options


class ResBlockBN(ResBlock):
    '''
    Residual block for Residual Network with batch normalization.

    Parameters
    ----------
    kernel_sizes : iter-of-ints, optional
        Kernel size of the convolution filters.
        Default: 3
    n_filters : iter-of-ints, optional
        List of numbers of filter in each convolution layers.
        Default: (16, 16)
    strides : iter-of-ints, optional
        List of stride in each convolution layers.
    batch_norm_first : bool, optional
        Specify whether to add batch normal layer before conv layer.
        Default: True

    Returns
    -------
    :class:`ResBlockBN`

    '''

    type_desc = 'Residual Block with BN'
    type = 'block'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None, batch_norm_first=True):
        number_of_instances = 0
        ResBlock.__init__(self, kernel_sizes, n_filters, strides, batch_norm_first)

    def add_layers(self):
        if self.batch_norm_first:
            for n_filter, kernel_size, stride in zip(self.n_filters, self.kernel_sizes, self.strides):
                self.layers.append(BN(act='relu'))
                self.layers.append(Conv2d(n_filters=n_filter, width=kernel_size, act='identity',
                                          stride=stride, include_bias=False))
        else:
            for n_filter, kernel_size, stride in zip(self.n_filters, self.kernel_sizes, self.strides):
                self.layers.append(Conv2d(n_filters=n_filter, width=kernel_size, act='identity',
                                          stride=stride, include_bias=False))
                self.layers.append(BN(act='relu'))

        self.layers.append(Res(act='identity'))

    def compile(self, src_layer, block_num=None):
        '''
        Convert the block structure into DLPy layer definitions.

        Parameters
        ----------
        src_layer : Layer
            The source layer for the whole block.
        block_num : int, optional
            The label of the block. (used to name the layers)

        Returns
        -------
        list
            A list of keyword-arguments

        '''

        if block_num is None:
            block_num = self.get_number_of_instances()

        options = []
        conv_num = 1
        bn_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.type == 'convo':
                layer.name = 'R{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]
            elif layer.type == 'batchnorm':
                layer.name = 'R{}B{}'.format(block_num, bn_num)
                bn_num += 1
                layer.src_layers = [input_layer]
            elif layer.type == 'residual':
                layer.name = 'Res{}'.format(block_num)
                layer.src_layers = [input_layer, src_layer]
            input_layer = layer
            options.append(layer.to_model_params())
        return options


class ResBlock_Caffe(ResBlock):
    '''
    Residual block for Residual Network with batch normalization.

    Parameters
    ----------
    kernel_sizes : iter-of-ints, optional
        Kernel size of the convolution filters.
        Default: 3
    n_filters : iter-of-ints, optional
        List of numbers of filter in each convolution layers.
        Default: (16, 16)
    strides : iter-of-ints, optional
        List of stride in each convolution layers.
    batch_norm_first : bool, optional
        Specify whether to add batch normal layer before conv layer.
        Default: False
    conv_short_cut : bool, optional
       Set to True, if there is short cut in the convolution layer
       Default: False

    Returns
    -------
    :class:`ResBlock_Caffe`

    '''

    type = 'block'
    type_desc = 'Residual Caffe Block '
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, kernel_sizes=3, n_filters=(16, 16), strides=None, batch_norm_first=False, conv_short_cut=False):
        number_of_instances = 0
        ResBlock.__init__(self, kernel_sizes, n_filters, strides, batch_norm_first, conv_short_cut)

    def add_layers(self):
        if self.batch_norm_first:
            if self.conv_short_cut:
                self.layers.append(BN(act='relu'))
                self.layers.append(Conv2d(n_filters=self.n_filters[-1], width=1, act='identity',
                                          stride=self.strides[0], include_bias=False))
            self.layers.append(Res(act='identity'))

            for n_filter, kernel_size, stride in zip(self.n_filters, self.kernel_sizes, self.strides):
                self.layers.append(BN(act='relu'))
                self.layers.append(Conv2d(n_filters=n_filter, width=kernel_size, act='identity',
                                          stride=stride, include_bias=False))
        else:
            if self.conv_short_cut:
                self.layers.append(
                    Conv2d(n_filters=self.n_filters[-1], width=1, act='identity',
                           stride=self.strides[0],include_bias=False))
                self.layers.append(BN(act='identity'))

            for n_filter, kernel_size, stride in zip(self.n_filters, self.kernel_sizes, self.strides):
                self.layers.append(Conv2d(n_filters=n_filter, width=kernel_size, act='identity',
                                          stride=stride, include_bias=False))
                self.layers.append(BN(act='relu'))

            self.layers.append(Res(act='relu'))

    def compile(self, src_layer, block_num=None):
        '''
        Compile the block structure into DLPy layer definitions.

        Parameters
        ----------
        src_layer : Layer
            The source layer for the whole block.
        block_num : int, optional
            The label of the block. (used to name the layers)

        Returns
        -------
        list
            A list of keyword-arguments

        '''

        if block_num is None:
            block_num = self.get_number_of_instances()

        options = []
        conv_num = 1
        bn_num = 1
        input_layer = src_layer
        if self.conv_short_cut:
            for layer in self.layers[:2]:
                if layer.type == 'convo':
                    layer.name = 'R{}C{}'.format(block_num, 0)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.type == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, 0)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                input_layer = layer
                options.append(layer.to_model_params())
            short_cut_layer = layer
            input_layer = src_layer
            for layer in self.layers[2:]:
                if layer.type == 'convo':
                    layer.name = 'R{}C{}'.format(block_num, conv_num)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.type == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, bn_num)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                elif layer.type == 'residual':
                    layer.name = 'Res{}'.format(block_num)
                    layer.src_layers = [input_layer, short_cut_layer]
                input_layer = layer
                options.append(layer.to_model_params())
        else:
            for layer in self.layers:
                if layer.type == 'convo':
                    layer.name = 'R{}C{}'.format(block_num, conv_num)
                    conv_num += 1
                    layer.src_layers = [input_layer]
                elif layer.type == 'batchnorm':
                    layer.name = 'R{}B{}'.format(block_num, bn_num)
                    bn_num += 1
                    layer.src_layers = [input_layer]
                elif layer.type == 'residual':
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
    n_cells : int, optional
        Number of cells
        Default: 4
    kernel_size : int, optional
        Size of the kernel
        Default: 3
    n_filter : int, optional
        Number of filters
        Default: 12
    stride : int, optional
        Size of the stride
        Default: 1

    Returns
    -------
    :class:`DenseNetBlock`

    '''

    type = 'block'
    type_desc = 'DenseNet Block '
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, n_cells=4, kernel_size=3, n_filter=12, stride=1):
        self.count_instances()
        self.config = dict()
        self.layers = []
        self.n_cells = n_cells
        self.kernel_size = kernel_size
        self.n_filter = n_filter
        self.stride = stride
        self.add_layers()

    # To keep track of the instance counts this will be using later to assing a name if not set
    number_of_instances = 0

    @classmethod
    def count_instances(cls):
        cls.number_of_instances += 1

    @classmethod
    def get_number_of_instances(cls):
        return cls.number_of_instances

    def add_layers(self):
        ''' Add layers for the block '''
        for _ in range(self.n_cells):
            self.layers.append(BN(act='relu'))
            self.layers.append(Conv2d(n_filters=self.n_filter, width=self.kernel_size, act='relu',
                                      stride=self.stride, include_bias=False))
            self.layers.append(Concat(act='identity'))

    def compile(self, src_layer, block_num=None):
        '''
        Convert the options into DLPy layer definition.

        Parameters
        ----------
        src_layer : Layer
            The source layer for the whole block.
        block_num : int, optional
            The label of the block. (used to name the layers)

        Returns
        -------
        dict
            A dictionary of keyword-arguments

        '''
        if block_num is None:
            block_num = self.get_number_of_instances()

        options = []
        conv_num = 1
        bn_num = 1
        concat_num = 1
        input_layer = src_layer
        for layer in self.layers:
            if layer.type == 'convo':
                layer.name = 'D{}C{}'.format(block_num, conv_num)
                conv_num += 1
                layer.src_layers = [input_layer]
            elif layer.type == 'batchnorm':
                layer.name = 'D{}B{}'.format(block_num, bn_num)
                bn_num += 1
                layer.src_layers = [input_layer]
            elif layer.type == 'concat':
                layer.name = 'D{}Concat{}'.format(block_num, concat_num)
                concat_num += 1
                layer.src_layers = [input_layer, src_layer]
                src_layer = layer
            input_layer = layer
            options.append(layer.to_model_params())
        return options


class Bidirectional(object):
    '''
    Bidirectional RNN layers

    Parameters
    ----------
    n : int or list of int
        Specifies the number of neurons in the recurrent layer. If n_blocks=1,
        then n should be an int. If n_blocks > 1, then n can be an int or a
        list of ints to indicate the number of neurons in each block.
    n_blocks : int, optional
        Specifies the number of bidirectional recurrent layer blocks.
        Default: 1
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Default: GRU
        Valid Values: RNN, LSTM, GRU
    output_type : string, optional
        Specifies the output type of the recurrent layer.
        Default: SAMELENGTH
        Valid Values: ENCODING, SAMELENGTH, ARBITRARYLENGTH
    max_output_length : int, mostly optional
        Specifies the maximum number of tokens to generate when the outputType
        parameter is set to ARBITRARYLENGTH.
    dropout : float, optional
        Specifies the dropout rate.
        Default: 0.2
    src_layers : list, optional
        Specifies the list of source layers for the layer.
    name : string, optional
        Specifies layer names. If not specified, 'RNN' is used

    Returns
    -------
    :class:`Bidirectional'

    '''

    type = 'block'

    def __init__(self, n, n_blocks=1, rnn_type='gru', output_type='samelength', dropout=0.2,
                 max_output_length=None, src_layers=None, name=None):

        if isinstance(n, int):
            if n_blocks == 1:
                self.n = [n]
            elif n_blocks > 1:
                self.n = [n] * n_blocks
            else:
                raise DLPyError('n_blocks should be larger than 0.')
        else:
            if len(n) == n_blocks:
                self.n = n
            else:
                raise DLPyError('the length of the neurons should be equal to the number of blocks')

        self.n_blocks = n_blocks
        self.src_layers = src_layers
        self.max_output_length = max_output_length
        self.rnn_type = rnn_type
        self.output_type = output_type
        self.dropout = dropout
        self.layers = []
        self.name = name
        self.add_layers()

    def add_layers(self):
        ''' Add layers for the block '''
        if self.src_layers is None:
            self.layers.append(InputLayer(name='input_layer_to_bidirectional_rnn'))

        for i in range(0, self.n_blocks):
            self.layers.append(Recurrent(n=self.n[i], rnn_type=self.rnn_type, output_type=self.output_type,
                                         dropout=self.dropout, reversed_=True,
                                         max_output_length=self.max_output_length))
            self.layers.append(Recurrent(n=self.n[i], rnn_type=self.rnn_type, output_type=self.output_type,
                                         dropout=self.dropout, reversed_=False,
                                         max_output_length=self.max_output_length))

    def get_last_layers(self):
        ''' Return last two layers, if they exist '''
        if len(self.layers) > 1:
            return self.layers[-2:]
        else:
            return None

    def compile(self, block_num=1):
        '''
        Convert the options into DLPy layer definition.

        Parameters
        ----------
        src_layer : Layer
            The source layer for the whole block.
        block_num : int, optional
            The label of the block. (Used to name the layers.)

        Returns
        -------
        dict
            A dictionary of keyword-arguments

        '''

        options = []
        if self.src_layers is None:
            input_layer = self.layers[0]
            i = 1
            options.append(input_layer.to_model_params())
        else:
            input_layer = self.src_layers
            i = 0

        local_name = 'RNN'
        bnum = block_num
        if self.name is not None:
            local_name = self.name
            bnum = 1

        while (i+1) < len(self.layers):
            layer1 = self.layers[i]
            layer1.name = local_name+'{}B{}'.format(0, bnum)
            if isinstance(input_layer, list):
                layer1.src_layers = input_layer
            else:
                layer1.src_layers = [input_layer]
            options.append(layer1.to_model_params())

            layer2 = self.layers[i+1]
            layer2.name = local_name+'{}B{}'.format(1, bnum)
            if isinstance(input_layer, list):
                layer2.src_layers = input_layer
            else:
                layer2.src_layers = [input_layer]
            options.append(layer2.to_model_params())
            input_layer = [layer1, layer2]
            bnum += 1
            i += 2

        return options

    def get_layers(self):
        ''' Return list of layers '''
        return self.layers
