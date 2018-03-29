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

''' Common layers for the deep learning models '''

from .utils import prod_without_none


class Layer(object):
    '''
    Base class for all layers

    Parameters
    ----------
    name : str
        Specifies the name of the layer.
    config : dict
        Specifies the configuration of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Layer`

    '''

    def __init__(self, name=None, config=None, src_layers=None):
        self.name = name
        self.config = config
        if isinstance(src_layers, list):
            self.src_layers = src_layers
        elif src_layers is None:
            self.src_layers = None
        else:
            self.src_layers = list(src_layers)

        if 'act' in self.config.keys():
            self.activation = config['act'].title()
        else:
            self.activation = None

        self.output_size = None
        self.kernel_size = None
        self.num_weights = None
        self.num_bias = None

    def to_model_params(self):
        '''
        Convert the model configuration to SAS Viya parameters

        Returns
        -------
        dict

        '''
        if self.config['type'].lower() == 'input':
            return dict(name=self.name, layer=self.config)
        else:
            return dict(name=self.name, layer=self.config,
                        srclayers=[item.name for item in self.src_layers])

    @property
    def summary_str(self):
        '''
        Generate the summary string describing the configuration of the layer.
        '''
        if self.config['type'].lower() == 'input':
            self.output_size = (int(self.config['width']),
                                int(self.config['height']),
                                int(self.config['nchannels']))
            self.kernel_size = None
            self.num_weights = 0
            self.num_bias = 0

        elif self.config['type'].lower() in ('convo', 'convolution'):
            self.output_size = (int(self.src_layers[0].output_size[0] //
                                    self.config['stride']),
                                int(self.src_layers[0].output_size[1] //
                                    self.config['stride']),
                                int(self.config['nfilters']))
            self.kernel_size = (int(self.config['width']), int(self.config['height']))
            self.num_weights = int(self.config['width'] *
                                   self.config['height'] *
                                   self.config['nfilters'] *
                                   self.src_layers[0].output_size[2])
            if 'includeBias' in self.config.keys():
                if self.config['includeBias'] is False:
                    self.num_bias = 0
                else:
                    self.num_bias = int(self.config['nfilters'])
            else:
                self.num_bias = int(self.config['nfilters'])

        elif self.config['type'].lower() in ('pool', 'pooling'):
            self.output_size = (int(self.src_layers[0].output_size[0] //
                                    self.config['stride']),
                                int(self.src_layers[0].output_size[1] //
                                    self.config['stride']),
                                int(self.src_layers[0].output_size[2]))
            self.kernel_size = (int(self.config['width']), int(self.config['height']))
            self.num_weights = 0
            self.num_bias = 0

        elif self.config['type'].lower() == 'batchnorm':
            self.output_size = self.src_layers[0].output_size
            self.kernel_size = None
            self.num_weights = 0
            self.num_bias = int(2 * self.src_layers[0].output_size[2])

        elif self.config['type'].lower() == 'residual':
            self.output_size = (int(min([item.output_size[0]
                                         for item in self.src_layers])),
                                int(min([item.output_size[1]
                                         for item in self.src_layers])),
                                int(max([item.output_size[2]
                                         for item in self.src_layers])))
            self.kernel_size = None
            self.num_weights = 0
            self.num_bias = 0

        elif self.config['type'].lower() == 'concat':
            self.output_size = (int(self.src_layers[0].output_size[0]),
                                int(self.src_layers[0].output_size[1]),
                                int(sum([item.output_size[2]
                                         for item in self.src_layers])))
            self.kernel_size = None
            self.num_weights = 0
            self.num_bias = 0

        elif self.config['type'].lower() in ('fc', 'fullconnect'):
            if isinstance(self.src_layers[0].output_size, int):
                num_features = self.src_layers[0].output_size
            else:
                num_features = prod_without_none(self.src_layers[0].output_size)
            self.output_size = int(self.config['n'])
            self.kernel_size = (int(num_features), int(self.config['n']))
            self.num_weights = int(num_features * self.config['n'])
            self.num_bias = int(self.config['n'])

        elif self.config['type'].lower() == 'output':
            self.type_name = 'Output'
            if isinstance(self.src_layers[0].output_size, int):
                num_features = self.src_layers[0].output_size
            else:
                num_features = prod_without_none(self.src_layers[0].output_size)

            if 'n' in self.config.keys():
                self.output_size = int(self.config['n'])
                self.kernel_size = (int(num_features), int(self.config['n']))
                self.num_weights = int(num_features * self.config['n'])
                self.num_bias = int(self.config['n'])
            else:
                self.kernel_size = None
                self.num_weights = None
                self.num_bias = None
                self.output_size = None

        name = '{}({})'.format(self.name, self.type_name)
        if len(name) > 17:
            col1 = '| {:<17}'.format('{}'.format(name[:14] + '...'))
        else:
            col1 = '| {:<17}'.format('{}'.format(name))

        col2 = '|{:^15}'.format('{}'.format(self.kernel_size))

        if 'stride' not in self.config.keys():
            col3 = '|{:^8}'.format('None')
        else:
            col3 = '|{:^8}'.format('{}'.format(int(self.config['stride'])))

        col4 = '|{:^12}'.format('{}'.format(self.activation))

        col5 = '|{:^17}'.format('{}'.format(self.output_size))

        num_paras = '{} / {}'.format(self.num_weights, self.num_bias)
        col6 = '|{:^22}|\n'.format(num_paras)

        return col1 + col2 + col3 + col4 + col5 + col6


class InputLayer(Layer):
    '''
    Input layer
    
    Parameters
    ----------
    n_channels : int
        Specifies the number of channels of the input images.
    width : int
        Specifies the width of the input images.
    height : int
        Specifies the height of the input images.
    scale : double, between 0 and 1.
        Specifies the scaling parameter of the image.
    dropout : double, between 0 and 1
        Specifies the dropout rate.
    offsets : iter-of-doubles
        Specifies the offset values.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`InputLayer`

    '''

    def __init__(self, n_channels=3, width=224, height=224, scale=1,
                 dropout=0, offsets=None, name=None, **kwargs):
        if offsets is None:
            offsets = [0] * n_channels
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'input'
        Layer.__init__(self, name, config)
        self.color_code = '#F0FF00'
        self.type_name = 'Input'


class Conv2d(Layer):
    '''
    2D convolutional layer

    Parameters
    ----------
    n_filters : int
        Specifies the number of filters.
    width : int
        Specifies the width of pooling window.
    height : int
        Specifies the height of pooling window.
    stride : int
        Specifies the step size of the moving window.
    act : str
        Specifies the activation types.
    dropout : double, between 0 and 1
        Specifies the dropout rate.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Conv2d`

    '''

    def __init__(self, n_filters, width=None, height=None, stride=1,
                 act='relu', dropout=0, name=None, src_layers=None, **kwargs):
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
        self.type_name = 'Convo.'


class Pooling(Layer):
    '''
    Pooling layer

    Parameters
    ----------
    width : int
        Specifies the width of pooling window.
    height : int
        Specifies the height of pooling window.
    stride : int
        Specifies the step size of the moving window.
    act : str
        Specifies the activation types.
    dropout : double, between 0 and 1
        Specifies the dropout rate.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Pooling`

    '''

    def __init__(self, width=None, height=None, stride=None,
                 pool='max', dropout=0, name=None, src_layers=None, **kwargs):
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
        self.activation = pool.title()
        self.color_code = '#FF9700'
        self.type_name = 'Pool'


class Dense(Layer):
    '''
    Fully connected layer

    Parameters
    ----------
    n : int
        Specifies the number of neurons.
    act : str
        Specifies the activation types.
    dropout : double, between 0 and 1
        Specifies the dropout rate.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Dense`

    '''

    def __init__(self, n, act='relu', dropout=0,
                 name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'fc'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#00ECFF'
        self.type_name = 'F.C.'


class Recurrent(Layer):
    '''
    Recurrent layer

    Parameters
    ----------
    n : int
        Specifies the number of neurons.
    act : str
        Specifies the activation types.
    rnn_type : str
        Specifies the type of RNN gate.
    output_type : str
        Specifies the type of output neurons.
    reversed_ : bool
        Specifies whether to reverse the sequence.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Recurrent`

    '''

    def __init__(self, n, act='AUTO', rnn_type='RNN', output_type='ENCODING',
                 reversed_=False, name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'recurrent'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FFA4A4'
        self.type_name = 'Rec.'


class BN(Layer):
    '''
    Batch Normalization layer

    Parameters
    ----------
    act : str
        Specifies the activation types.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`BN`

    '''

    def __init__(self, act='AUTO', name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'batchnorm'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FFF999'
        self.type_name = 'B.N.'


class Res(Layer):
    '''
    Residual layer

    Parameters
    ----------
    act : str
        Specifies the activation types.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional.
        Specifies the source layer(s).

    Returns
    -------
    :class:`Res`

    '''

    def __init__(self, act='AUTO', name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'residual'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FF0000'
        self.type_name = 'Resid.'


class Concat(Layer):
    '''
    Concatenation layer

    Parameters
    ----------
    act : str
        Specifies the activation types.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Concat`

    '''

    def __init__(self, act='AUTO', name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'concat'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#DD5022'
        self.type_name = 'Concat.'


class Proj(Layer):
    '''
    Projection layer

    Parameters
    ----------
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Proj`

    '''

    def __init__(self, name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'projection'
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#FFA2A3'
        self.type_name = 'Proj.'


class OutputLayer(Layer):
    '''
    Output layer

    Parameters
    ----------
    n : int
        Specifies the number of output neurons.
    act : str
        Specifies the activation types.
    name : str
        Specifies the name of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`OutputLayer`
 
    '''

    def __init__(self, n=None, act='softmax', name=None, src_layers=None, **kwargs):
        config = locals()
        config = _unpack_config(config)
        config['type'] = 'output'
        if config['n'] is None:
            del config['n']
        Layer.__init__(self, name, config, src_layers)
        self.color_code = '#C8C8C8'
        self.type_name = 'Output.'


def _unpack_config(config):
    ''' Unpack the configuration from the keyword-argument-only input '''
    kwargs = config['kwargs']
    del config['self'], config['name'], config['kwargs']
    try:
        del config['src_layers']
    except:
        pass
    out = {}
    out.update(config)
    out.update(kwargs)
    for key in out:
        if '_' in key:
            new_key = key.replace('_', '')
            out[new_key] = out[key]
            out.pop(key)
    return out
