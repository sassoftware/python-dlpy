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

''' Sequential object for deep learning '''

from .model import Model


class Sequential(Model):
    '''
    Model for sequentially building of deep learning models

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    layers : list of Layer objects.
        Specifies the layers of the sequential model.
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default : None

    Returns
    -------
    :class:`Sequential`

    '''

    def __init__(self, conn, layers=None, model_table=None):
        Model.__init__(self, conn, model_table=model_table)

        if layers is None:
            self.layers = []
        elif type(layers) is not dict:
            raise TypeError('layers has to be a list of layer(s).')
        else:
            self.layers = layers
            if layers[-1]['type'] == 'output':
                self.compile()

    def add(self, layer):
        '''
        Add layer(s) to model

        Parameters
        ----------
        layer : Layer or list-of-Layer
            Specifies the layer to be added

        '''
        if self.layers == [] and layer.config['type'].lower() != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if len(self.layers) > 0 and layer.config['type'] is 'input':
            raise ValueError('Only the first layer of the Sequential model '
                             'can be an input layer')
        self.layers.append(layer)

        if layer.config['type'].lower() == 'input':
            print('NOTE: Input layer added.')

        elif layer.config['type'].lower() in ('convo', 'convolution'):
            print('NOTE: Convolutional layer added.')

        elif layer.config['type'].lower() in ('pool', 'pooling'):
            print('NOTE: Pooling layer added.')

        elif layer.config['type'].lower() in ('fc', 'fullconnect'):
            print('NOTE: Fully-connected layer added.')

        elif layer.config['type'].lower() == 'batchnorm':
            print('NOTE: Batch Normalization layer added.')

        elif layer.config['type'].lower() == 'block':
            print('NOTE: A block of layers added.')

        elif layer.config['type'].lower() == 'output':
            print('NOTE: Output layer added.')
            self.compile()

    def pop(self, loc=-1):
        '''
        Delete layer(s) from model and return it

        Parameters
        ----------
        loc : int
            Specifies the index of the layer in the model

        '''
        if len(self.layers) > 0:
            return self.layers.pop(loc)

    def switch(self, loc1, loc2):
        '''
        Switch the oder of two layers in the model.

        Parameters
        ----------
        loc1 : int
            Specifies the index of the first layer
        loc2 : int
            Specifies the index of the second layer

        '''
        self.layers[loc1], self.layers[loc2] = self.layers[loc2], self.layers[loc1]

    def compile(self):
        '''
        Convert the layer objects into Viya options

        '''
        if self.layers[0].config['type'] != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        # if self.layers[-1].config['type'] != 'output':
        #     raise ValueError('The last layer of the model must be an output layer')
        self._retrieve_('deeplearn.buildmodel', model=dict(
            name=self.model_name, replace=True), type='CNN')

        conv_num = 1
        fc_num = 1
        bn_num = 1
        block_num = 1

        compiled_layers = []
        output_layer = None

        for layer in self.layers:
            if layer.config['type'] == 'block':
                options = layer.compile(src_layer=output_layer, block_num=block_num)
                block_num += 1
                for item in layer.layers:
                    compiled_layers.append(item)
                output_layer = layer.layers[-1]
                for option in options:
                    self._retrieve_('deeplearn.addlayer', model=self.model_name, **option)
            else:
                # Name each layer of the model.
                if layer.config['type'] == 'input':
                    if layer.name is None:
                        layer.name = 'Data'
                else:
                    layer.src_layers = [output_layer]
                    if layer.config['type'].lower() in ('convo', 'convolution'):
                        if layer.name is None:
                            layer.name = 'Conv{}_{}'.format(block_num, conv_num)
                            conv_num += 1
                    elif layer.config['type'].lower() == 'batchnorm':
                        if layer.name is None:
                            layer.name = 'BN{}_{}'.format(block_num, bn_num)
                            bn_num += 1
                    elif layer.config['type'].lower() in ('pool', 'pooling'):
                        if layer.name is None:
                            layer.name = 'Pool{}'.format(block_num)
                            block_num += 1
                            conv_num = 1
                    elif layer.config['type'].lower() in ('fc', 'fullconnect'):
                        if layer.name is None:
                            layer.name = 'FC{}'.format(fc_num)
                            fc_num += 1
                    elif layer.config['type'].lower() == 'output':
                        if layer.name is None:
                            layer.name = 'Output'
                    else:
                        raise ValueError('{} is not a supported layer type'.format(
                            layer.config['type']))

                option = layer.to_model_params()
                compiled_layers.append(layer)
                output_layer = layer

                self._retrieve_('deeplearn.addlayer', model=self.model_name, **option)

        print('NOTE: Model compiled successfully.')
        self.layers = compiled_layers
