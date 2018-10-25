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

import pandas as pd
import six
from dlpy.utils import multiply_elements, DLPyError, camelcase_to_underscore, underscore_to_camelcase
from . import __dev__
import warnings

PALETTES = dict(
    original={
        'input': '#F0FF00',
        'convo': '#6CFF00',
        'pool': '#FF9700',
        'fc': '#00ECFF',
        'recurrent': '#FFA4A4',
        'batchnorm': '#FFF999',
        'residual': '#FF0000',
        'concat': '#DD5022',
        'projection': '#FFA2A3',
        'output': '#C8C8C8',
        'keypoints': '#C8C8C8',
        'detection': '#C8C8C8',
        'scale': '#C8C8C8',
        'fcmp': '#C8C8C8',
        'reshape': '#C8C8C8',
        'unknown': '#FFFFFF',
    },
    default={
        'input': '#3288bd40',  # blue
        'convo': ('#b58c15', '#fee08b40'),  # flesh
        'pool': '#66c2a540',  # teal
        'fc': ('#aeae82', '#ffffbf40'),  # yellow
        'recurrent': '#abdda440',  # mint
        'batchnorm': '#fdae6140',  # tangerine
        'residual': '#e6f59840',  # lime
        'concat': '#f46d4340',  # orange
        'projection': '#d53e4f40',  # rose
        'output': '#5e4fa220',  # purple
        'keypoints': '#5e4fa220',  # purple
        'detection': '#5e4fa220',  # purple
        'scale': '#5e4fa220',  # purple
        'fcmp': '#5e4fa220',  # purple
        'reshape': '#5e4fa220',  # purple
        'unknown': '#9e014240',  # crimson
    }
)


def get_color(name, palette='default'):
    '''
    Get specified color name from palette

    Parameters
    ----------
    name : string
        The name of the color
    palette : string, optional
        The pallete to get the color from

    Returns
    -------
    string
        Hex encoded color

    '''
    name = name.strip().lower().replace('.', '')
    return PALETTES[palette].get(name, PALETTES[palette]['unknown'])


class Layer(object):
    '''
    Base class for all layers

    Parameters
    ----------
    name : str, optional
        Specifies the name of the layer.
    config : dict, optional
        Specifies the configuration of the layer.
    src_layers : iter-of-Layers, optional
        Specifies the source layer(s).

    Returns
    -------
    :class:`Layer`

    '''

    type = 'layer'
    type_label = 'Layer'
    type_desc = 'Base layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, config=None, src_layers=None):
        self.name = name
        self.config = config

        if src_layers is None:
            self.src_layers = None
        else:
            self.src_layers = list(src_layers)

        if 'act' in self.config.keys() and self.config['act'] is not None:
            self.activation = self.config['act'].title()
        else:
            self.activation = None

    @classmethod
    def count_instances(cls):
        cls.number_of_instances += 1

    @classmethod
    def get_number_of_instances(cls):
        return cls.number_of_instances

    def format_name(self, block_num=None, local_count=None):
        '''
        Format the name of the layer

        This function will be called from sequential.

        Parameters
        ----------
        block_num : int, optional
            Block number
        local_count : int, optional
            Number of instances

        '''
        if block_num is None and local_count is None:
            self.count_instances()
            self.name = self.type_label+'{}'.format(self.get_number_of_instances())
        elif block_num is None:
            self.name = self.type_label+'{}'.format(local_count)
        elif local_count is None:
            self.count_instances()
            self.name = self.type_label+'{}_{}'.format(block_num, type(self).get_number_of_instances())
        else:
            self.name = self.type_label+'{}_{}'.format(block_num, local_count)

    def to_model_params(self):
        '''
        Convert the model configuration to CAS action parameters

        Returns
        -------
        dict

        '''
        new_params = {}
        for key, value in six.iteritems(self.config):
            if value is not None:
                new_key = key.replace('_', '')
                if isinstance(new_key, str):
                    new_key = new_key.lower()
                if isinstance(value, str):
                    value = value.lower()
                new_params[new_key] = value
        new_params['type'] = self.type
        if self.type == 'input':
            return dict(name=self.name, layer=new_params)
        else:
            return dict(name=self.name, layer=new_params,
                        srclayers=[item.name for item in self.src_layers])

    @property
    def summary(self):
        ''' Return a DataFrame containing the layer information '''
        return pd.DataFrame([[self.name, self.type, self.kernel_size,
                              self.config.get('stride', None), self.activation,
                              self.output_size, (self.num_weights, self.num_bias)]],
                            columns=['Layer', 'Type', 'Kernel Size', 'Stride',
                                     'Activation', 'Output Size',
                                     'Number of Parameters'])

    @property
    def rnn_summary(self):
        ''' Return a DataFrame containing the layer information for rnn models'''
        return pd.DataFrame([[self.name, self.type, self.activation, self.output_size]],
                            columns=['Layer', 'Type', 'Activation', 'Output Size'])


class InputLayer(Layer):
    '''
    Input layer

    Parameters
    ----------
    name : string
        Specifies the name of the input layer.
    nominals : string-list, optional
        Specifies the nominal input variables to use in the analysis.
    std : string, optional
        Specifies how to standardize the variables in the input layer.
        Valid Values: MIDRANGE, NONE, STD
    n_channels : int, optional for rnn required for cnn
        Specifies the depth of the input data, used if data is image.
    width : int, optional for rnn required for cnn
        Specifies the width of the input data, used if data is image.
    height : int, optional
        Specifies the height of the input data, used if data is image.
        Note: Required for CNN.
    scale : float, optional,
        Specifies the scale to be used to scale the input data.
    offsets: int-list, optional
        Specifies the values to be subtracted from the pixel values of
        the input data, used if the data is image.
    dropout : float, optional
        Specifies the dropout rate.
    random_flip : int, optional
        Specifies the type of the random flip to be applied to the input
        data, used if data is image.
        Valid Values: NONE, H, V, HV
        Default: NONE
    random_crop : int, optional
        Specifies the type of the random crop to be applied to the input data, used if data is image.
        Valid Values: NONE, UNIQUE
        Default: NONE
    random_mutation : int, optional
        Specifies the type of the random mutation to be applied to the input data, used if data is image.
        Valid Values: NONE, RANDOM
        Default: NONE

    Returns
    -------
    :class:`InputLayer`

    '''

    type = 'input'
    type_label = 'Input'
    type_desc = 'Input layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, n_channels=None, width=None, height=None, name=None, nominals=None, std=None, scale=None,
                 offsets=None, dropout=None, random_flip=None, random_crop=None, random_mutation=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        if offsets is None and n_channels is not None:
            offsets = [0] * n_channels
        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_input_parameters(parameters)

        if width is None and height is not None:
            parameters['width'] = height
        if height is None and width is not None:
            parameters['height'] = width

        number_of_instances = 0
        Layer.__init__(self, name, parameters)
        if n_channels is not None and (width or height is not None):
            self._output_size = (int(self.config['width']), int(self.config['height']), int(self.config['n_channels']))
        else:
            self._output_size = 0

        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        if self.config['width'] is not None:
            return self._output_size
        else:
            return 0

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def num_bias(self):
        return 0


class Conv2d(Layer):
    '''
    Convolution layer

    Parameters
    ----------

    n_filters : int
        Specifies the number of filters for the layer.
    width : int
        Specifies the width of the kernel.
    height : int
        Specifies the height of the kernel.
    stride : int, optional
        Specifies the step size for the moving window of the kernel over the input data.
    name : string, optional
        Specifies the name of the convolution layer.
    stride_horizontal : int, optional
        Specifies the horizontal stride.
    stride_vertical : int, optional
        Specifies the vertical stride.
    padding : int, optional
        Specifies the padding size, assuming equal padding vertically and horizontally.
    padding_width : int, optional
        Specifies the length of the horizontal padding.
    padding_height : int, optional
        Specifies the length of the vertical padding.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init`` parameter is set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the ``init`` parameter is set to NORMAL
    init_bias : float, optional
        Specifies the initial bias for the layer.
    dropout : float, optional
        Specifies the dropout rate.
        Default: 0
    include_bias : bool, optional
        Includes bias neurons (default).
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Conv2d`

    '''

    type = 'convo'
    type_label = 'Convo.'
    type_desc = 'Convolution layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, n_filters, width=None, height=None, stride=1, name=None, stride_horizontal=None,
                 stride_vertical=None, padding=None, padding_width=None, padding_height=None, act='relu',
                 fcmp_act=None, init=None, std=None, mean=None, truncation_factor=None, init_bias=None,
                 dropout=None, include_bias=True, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        if width is None and height is None:
            parameters['width'] = 3
            parameters['height'] = 3
        elif width is None:
            parameters['width'] = height
        elif height is None:
            parameters['height'] = width
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)

        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self._num_weights = None
        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        if self._output_size is None:
            # calculate output according to specified padding
            if self.config['padding'] is not None:
                out_w = (self.src_layers[0].output_size[0] - 
                         self.config['width'] + 2*self.config['padding']) // self.config['stride'] + 1
                out_h = (self.src_layers[0].output_size[1] - 
                         self.config['height'] + 2*self.config['padding']) // self.config['stride'] + 1
            else:
                import math
                # same padding
                out_w = math.ceil(self.src_layers[0].output_size[0] / self.config['stride'])
                out_h = math.ceil(self.src_layers[0].output_size[1] / self.config['stride'])

                # if either padding_height or padding_width are specified
                if self.config['padding_width'] is not None:
                    out_w = (self.src_layers[0].output_size[0] - 
                             self.config['width'] + 
                             2*self.config['padding_width']) // self.config['stride'] + 1
                if self.config['padding_height'] is not None:
                    out_h = (self.src_layers[0].output_size[1] - 
                             self.config['height'] + 
                             2*self.config['padding_height']) // self.config['stride'] + 1
            self._output_size = (int(out_w), int(out_h), int(self.config['n_filters']))
        return self._output_size

    @property
    def num_weights(self):
        if self._num_weights is None:
            self._num_weights = int(self.config['width'] * self.config['height'] *
                                    self.config['n_filters'] * self.src_layers[0].output_size[2])
        return self._num_weights

    @property
    def kernel_size(self):
        return (int(self.config['width']), int(self.config['height']))

    @property
    def num_bias(self):
        if 'include_bias' in self.config:
            if not self.config['include_bias']:
                return 0
            else:
                return int(self.config['n_filters'])
        return int(self.config['n_filters'])


class Pooling(Layer):
    '''
    Pooling layer

    Parameters
    ----------
    width : int
        Specifies the width of the pooling window.
    height : int
        Specifies the height of the pooling window.
    name : string, optional
        Specifies the name of the layer.
    stride : int, optional
        Specifies the step size of the moving window, assuming a equal moves
        vertically and horizontally
    stride_horizontal : int, optional
        Specifies the step size of the moving window horizontally.
    stride_vertical : int, optional
        Specifies the step size of the moving window vertically.
    padding : int, optional
        Specifies the length of the padding assuming equal padding horizontally
        and vertically.
    padding_width : int, optional
        Specifies the length of the padding horizontally.
    padding_height : int, optional
        Specifies the length of the padding vertically.
    pool : string, optional
        Specifies the type of the pooling layer.
        Valid Values: MAX, MIN, MEAN, AVERAGE, FIXED, RANDOM, MEDIAN
        Default: MAX
    dropout : float, optional
        Specifies the dropout rate.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Pooling`

    '''

    type = 'pool'
    type_label = 'Pool'
    type_desc = 'Pooling layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, width=None, height=None, stride=None, name=None, stride_horizontal=None, stride_vertical=None,
                 padding=None, padding_width=None, padding_height=None, pool='max', dropout=None, src_layers=None,
                 **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        if width is None and height is None:
            parameters['width'] = 2
            parameters['height'] = 2
        elif width is None:
            parameters['width'] = height
        elif height is None:
            parameters['height'] = width

        if stride is None:
            parameters['stride'] = parameters['width']
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)

        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self.activation = pool.title()
        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        if self._output_size is None:
            # calculate output according to specified padding
            if self.config['padding'] is not None:
                out_w = (self.src_layers[0].output_size[0] - 
                         self.config['width'] + 2*self.config['padding']) // self.config['stride'] + 1
                out_h = (self.src_layers[0].output_size[1] - 
                         self.config['height'] + 2*self.config['padding']) // self.config['stride'] + 1
            else:
                import math
                # same padding
                out_w = math.ceil(self.src_layers[0].output_size[0] / self.config['stride'])
                out_h = math.ceil(self.src_layers[0].output_size[1] / self.config['stride'])

                # if either padding_height or padding_width are specified
                if self.config['padding_width'] is not None:
                    out_w = (self.src_layers[0].output_size[0] - 
                             self.config['width'] + 
                             2*self.config['padding_width']) // self.config['stride'] + 1
                if self.config['padding_height'] is not None:
                    out_h = (self.src_layers[0].output_size[1] - 
                             self.config['height'] + 
                             2*self.config['padding_height']) // self.config['stride'] + 1
            self._output_size = (int(out_w), int(out_h), int(self.src_layers[0].output_size[2]))
        return self._output_size

    @property
    def kernel_size(self):
        return (int(self.config['width']), int(self.config['height']))

    @property
    def num_weights(self):
        return 0

    @property
    def num_bias(self):
        return 0


class Dense(Layer):
    '''
    Fully connected layer

    Parameters
    ----------
    n : int
        Specifies the number of neurons.
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init`` parameter is
        set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the
        ``init`` parameter is set to NORMAL
    init_bias : float, optional
        Specifies the initial bias for the layer.
    dropout : float, optional
        Specifies the dropout rate.
        Default: 0
    include_bias : bool, optional
        Includes bias neurons (default).
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Dense`

    '''

    type = 'fc'
    type_label = 'F.C.'
    type_desc = 'Fully-connected layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, n, name=None, act='relu', fcmp_act=None, init=None, std=None, mean=None, truncation_factor=None,
                 init_bias=None, dropout=None, include_bias=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._num_features = None
        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        return int(self.config['n'])

    @property
    def num_bias(self):
        if 'include_bias' in self.config:
            if not self.config['include_bias']:
                return 0
            else:
                return int(self.config['n'])
        return int(self.config['n'])

    @property
    def num_features(self):
        if self._num_features is None:
            if self.src_layers is None:
                return 0
            if isinstance(self.src_layers[0].output_size, int):
                return self.src_layers[0].output_size
            self._num_features = multiply_elements(self.src_layers[0].output_size)
        return self._num_features

    @property
    def kernel_size(self):
        return (int(self.num_features), int(self.config['n']))

    @property
    def num_weights(self):
        return int(self.num_features * self.config['n'])


class Recurrent(Layer):
    '''
    RNN layer

    Parameters
    ----------
    n : int
        Specifies the number of neurons.
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init`` parameter is set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the
        ``init`` parameter is set to NORMAL
    rnn_type : string, optional
        Specifies the type of the rnn layer.
        Possible Values: RNN, LSTM, GRU
        Default: RNN
    output_type : string, optional
        Specifies the output type of the recurrent layer.
        Valid Values: ENCODING, SAMELENGTH, ARBITRARYLENGTH
        Default: ENCODING
    max_output_length : int, optional
        Specifies the maximum number of tokens to generate when the outputType
        parameter is set to ``ARBITRARYLENGTH``.
    reversed : bool, optional
        Specifies the direction of the rnn layer.
        Default: False
    dropout : float, optional
        Specifies the dropout rate.
        Default: 0
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Recurrent`

    '''

    type = 'recurrent'
    type_label = 'Rec.'
    type_desc = 'Recurrent layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, n, name=None, act='AUTO', fcmp_act=None, init=None, std=None, mean=None, truncation_factor=None,
                 rnn_type='LSTM', output_type='ENCODING', max_output_length=None, reversed_=None, dropout=None,
                 src_layers=None):
        parameters = locals()
        _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        return int(self.config['n'])

    @property
    def num_bias(self):
        if 'include_bias' in self.config:
            if not self.config['include_bias']:
                return 0
            else:
                return int(self.config['n'])
        return int(self.config['n'])

    @property
    def num_features(self):
        if self.src_layers is None:
            return 0
        if isinstance(self.src_layers[0].output_size, int):
            return self.src_layers[0].output_size
        return multiply_elements(self.src_layers[0].output_size)

    @property
    def kernel_size(self):
        return (int(self.num_features), int(self.config['n']))

    @property
    def num_weights(self):
        return int(self.num_features * self.config['n'])


class BN(Layer):
    '''
    Batch normalization layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`BN`

    '''
    type = 'batchnorm'
    type_label = 'B.N.'
    type_desc = 'Batch normalization layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self._num_bias = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            if self.src_layers is None:
                self._output_size = 0
            self._output_size = self.src_layers[0].output_size
        return self._output_size

    @property
    def num_bias(self):
        if self._num_bias is None:
            if self.src_layers is None:
                self._num_bias = 0
            self._num_bias = int(2 * self.src_layers[0].output_size[2])
        return self._num_bias


class Res(Layer):
    '''
    Residual layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Res`

    '''
    type = 'residual'
    type_label = 'Resid.'
    type_desc = 'Residual layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def num_bias(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            if self.src_layers is None:
                self._output_size = (0, 0, 0)
            self._output_size = (int(min([item.output_size[0] for item in self.src_layers])),
                                 int(min([item.output_size[1] for item in self.src_layers])),
                                 int(max([item.output_size[2] for item in self.src_layers])))
        return self._output_size


class Concat(Layer):
    '''
    Concat layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Concat`

    '''
    type = 'concat'
    type_label = 'Concat.'
    type_desc = 'Concatenation layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def num_bias(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            if self.src_layers is None:
                self._output_size = (0, 0, 0)
            self._output_size = (int(self.src_layers[0].output_size[0]),
                                 int(self.src_layers[0].output_size[1]),
                                 int(sum([item.output_size[2] for item in self.src_layers])))
        return self._output_size


class Proj(Layer):
    '''
    Projection layer

    Parameters
    ----------
    embedding_size : int
        Specifies the size of the embedding.
    alphabet_size : int
        Specifies the size of the alphabet.
    name : string, optional
        Specifies the name of the layer.
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init``
        parameter is set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the
        ``init`` parameter is set to NORMAL
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Proj`

    '''
    type = 'projection'
    type_label = 'Proj.'
    type_desc = 'Projection layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, embedding_size, alphabet_size, name=None, init=None,
                 std=None, mean=None, truncation_factor=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self.color_code = get_color(self.type)


class OutputLayer(Layer):
    '''
    Output layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    fcmp_error : string, optional
        Specifies the FCMP error function for the output layer.
    error : int, optional
        Specifies the error function. This function is also known as
        the loss function.
        Valid Values: AUTO, GAMMA, NORMAL, POISSON, ENTROPY, CTC, FCMPERR, CTC_ALT
        Default: AUTO
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init`` parameter is set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the ``init``
        parameter is set to NORMAL
    init_bias : float, optional
        Specifies the initial bias for the layer.
    n : int, optional
        Specifies the number of neurons in the output layer. By default, the number of
        neurons is automatically determined when the model training begins. The
        specified value cannot be smaller than the number of target variable levels.
    n_softmax_samples : int, optional
        Specifies the number of samples used in sampled Softmax.
    include_bias : bool, optional
        Includes bias neurons. Default: True
    target_std : int, optional
        Specifies how to standardize the variables in the output layer.
        Valid Values: MIDRANGE, NONE, STD
        Default: NONE
    full_connect : bool, optional
        In default, the output layer is fully connected to all the previous layers.
        When it is false, the output layer becomes a loss function layer.
        Default: True
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`OutputLayer`

    '''
    type = 'output'
    type_label = 'Output'
    type_desc = 'Output layer'
    can_be_last_layer = True
    number_of_instances = 0

    def __init__(self, name=None, act='softmax', fcmp_act=None, fcmp_error=None, error=None, init=None, std=None,
                 mean=None, truncation_factor=None, init_bias=None, n=None, n_softmax_samples=None, include_bias=None,
                 target_std=None, full_connect=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)

        if parameters['n'] is None:
            del parameters['n']

        Layer.__init__(self, name, parameters, src_layers)
        self._num_features = None
        self.color_code = get_color(self.type)

    @property
    def output_size(self):
        if 'n' in self.config:
            return int(self.config['n'])
        return None

    @property
    def num_bias(self):
        if self.config['full_connect'] == False:
            return 0
        else:
            if self.config['include_bias'] == False:
                return 0
            else:
                if 'n' in self.config:
                    return int(self.config['n'])
                return None

    @property
    def num_features(self):
        if self._num_features is None:
            if self.src_layers is None:
                self._num_features = 0
            elif isinstance(self.src_layers[0].output_size, int):
                self._num_features = self.src_layers[0].output_size
            else:
                self._num_features = multiply_elements(self.src_layers[0].output_size)
        return self._num_features

    @property
    def kernel_size(self):
        if 'n' not in self.config:
            return None
        if not self.config['full_connect']:
            return (int(self.num_features), int(self.config['n']))

    @property
    def num_weights(self):
        if self.config['full_connect'] == False:
            return 0
        else:
            if 'n' not in self.config:
                return None
            return int(self.num_features * self.config['n'])


class Keypoints(Layer):
    '''
    Keypoints layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    init : string, optional
        Specifies the initialization scheme for the layer.
        Valid Values: XAVIER, UNIFORM, NORMAL, CAUCHY, XAVIER1, XAVIER2, MSRA, MSRA1, MSRA2
        Default: XAVIER
    std : float, optional
        Specifies the standard deviation value when the ``init`` parameter
        is set to NORMAL.
    mean : float, optional
        Specifies the mean value when the ``init`` parameter is set to NORMAL.
    truncation_factor : float, optional
        Specifies the truncation threshold (truncationFactor x std), when the
        ``init`` parameter is set to NORMAL
    init_bias : float, optional
        Specifies the initial bias for the layer.
    n : int, optional
        Specifies the number of neurons in the output layer. By default, the number
        of neurons is automatically determined when the model training begins. The
        specified value cannot be smaller than the number of target variable levels.
        Default: 0
    include_bias : bool, optional
        Includes bias neurons (default).
    target_std : int, optional
        Specifies how to standardize the variables in the output layer.
        Valid Values: MIDRANGE, NONE, STD
        Default: NONE
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Keypoints`

    '''
    type = 'keypoints'
    type_label = 'K.P.'
    type_desc = 'Keypoints layer'
    can_be_last_layer = True
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, init=None, std=None, mean=None, truncation_factor=None,
                 init_bias=None, n=0, include_bias=None, target_std=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._num_features = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return (int(self.num_features), int(self.config['n']))

    @property
    def num_weights(self):
        return int(self.num_features * self.config['n'])

    @property
    def num_features(self):
        if self._num_features is None:
            if self.src_layers is None:
                self._num_features = 0
            elif isinstance(self.src_layers[0].output_size, int):
                self._num_features = self.src_layers[0].output_size
            else:
                self._num_features = multiply_elements(self.src_layers[0].output_size)
        return self._num_features

    @property
    def output_size(self):
        return int(self.config['n'])

    @property
    def num_bias(self):
        if self.config['include_bias'] == False:
            return 0
        else:
            return int(self.config['n'])


class Detection(Layer):
    '''
    Detection layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    detection_model_type : int, optional
        Specifies the type of the object detection model.
        Valid Values: YOLOV1, YOLOV2
        Default: YOLOV2
    anchors : iter-of-floats, optional
        Specifies the anchor box values. Anchor box values are a list of scalar
        value pairs that represent the normalized box sizes in X and Y
        direction for objects to be detected. The normalized box sizes
        are calculated by dividing the box size in pixels by the grid size.
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : int, optional
        Specifies the coordinates format type in the input label and detection result.
        Valid Values: RECT, COCO, YOLO
        Default: RECT
    class_number : int
        Specifies the number of classes to detection in the detection layer.
    grid_number : int
        Specifies the number of grids per side in the detection layer.
    predictions_per_grid : int
        Specifies the number of predictions to generate in the detection layer.
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Detection`

    '''
    type = 'detection'
    type_label = 'Detection'
    type_desc = 'Detection layer'
    can_be_last_layer = True
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', detection_model_type=None, anchors=None, softmax_for_class_prob=None,
                 coord_type=None, class_number=None, grid_number=None, predictions_per_grid=None, do_sqrt=None,
                 coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                 detection_threshold=None, iou_threshold=None, random_boxes=None, src_layers=None, max_boxes=None,
                 max_label_per_image=None, match_anchor_size=None, num_to_force_coord=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            self._output_size = self.src_layers[0].output_size
        return self._output_size

    @property
    def num_bias(self):
        return 0


class Scale(Layer):
    '''
    Scale layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Scale`

    '''
    type = 'scale'
    type_label = 'Scale'
    type_desc = 'Scale layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, src_layers=None, **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        self._output_size = None
        Layer.__init__(self, name, parameters, src_layers)
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            self._output_size = (int(max(self.src_layers[0].output_size[0], self.src_layers[0].output_size[0])),
                                 int(max(self.src_layers[0].output_size[1], self.src_layers[0].output_size[1])),
                                 int(max(self.src_layers[0].output_size[2], self.src_layers[0].output_size[2])))
        return self._output_size

    @property
    def num_bias(self):
        return 0


class Reshape(Layer):
    '''
    Reshape layer

    Parameters
    ----------
    name : string, optional
        Specifies the name of the layer.
    act : string, optional
        Specifies the activation function.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    fcmp_act : string, optional
        Specifies the FCMP activation function for the layer.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.
    height : int, optional
        Specifies the height of the input data. By default the height
        is determined automatically when the model training begins.
    width : int, optional
        Specifies the width of the input data. By default the width
        is determined automatically when the model training begins.
    depth : int, optional
        Specifies the depth of the feature maps.
    src_layers : iter-of-Layers, optional
        Specifies the layers directed to this layer.

    Returns
    -------
    :class:`Reshape`

    '''
    type = 'reshape'
    type_label = 'Reshape'
    type_desc = 'Reshape layer'
    can_be_last_layer = False
    number_of_instances = 0

    def __init__(self, name=None, act='AUTO', fcmp_act=None, width=None, height=None, depth=None, src_layers=None,
                 **kwargs):

        if not __dev__ and len(kwargs) > 0:
            raise DLPyError('**kwargs can be used only in development mode.')

        parameters = locals()
        parameters = _unpack_config(parameters)
        # _clean_parameters(parameters)
        Layer.__init__(self, name, parameters, src_layers)
        self._output_size = None
        self.color_code = get_color(self.type)

    @property
    def kernel_size(self):
        return None

    @property
    def num_weights(self):
        return 0

    @property
    def output_size(self):
        if self._output_size is None:
            self._output_size = (self.config['width'], self.config['height'], self.config['depth'])
        return self._output_size

    @property
    def num_bias(self):
        return 0


def _clean_input_parameters(parameters):
    del parameters['self']
    del parameters['name']


def _clean_parameters(parameters):
    del parameters['src_layers']
    _clean_input_parameters(parameters)


def _unpack_config(config):
    ''' Unpack the configuration from the keyword-argument-only input '''
    kwargs = config['kwargs']
    new_kwargs = {}
    for key, value in six.iteritems(kwargs):
        new_kwargs[camelcase_to_underscore(key)] = value
    del config['self'], config['name'], config['kwargs']
    try:
        del config['src_layers']
    except:
        pass
    out = {}
    conflict_arg = [i for i in config if i in new_kwargs.keys()]
    for arg in conflict_arg:
        underscore_arg = underscore_to_camelcase(arg)
        warnings.warn('Since {0} and {1} specify the same parameter, {0} is ignored'.format(arg, underscore_arg))

    out.update(config)
    out.update(new_kwargs)
    # out.update(kwargs)
    return out
