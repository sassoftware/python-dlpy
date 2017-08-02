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
RNN Model object for deep learning.
'''

from .model import Model


class RNN(Model):
    def __init__(self, conn, layers=None, model_name=None):
        if not conn.queryactionset('deepLearn')['deepLearn']:
            conn.loadactionset('deepLearn')

        Model.__init__(self, conn, model_name=model_name)

        if layers is None:
            self.layers = []
        elif type(layers) is not dict:
            raise TypeError('layers has to be a list of layer(s).')
        else:
            self.layers = layers
            if layers[-1]['type'] == 'output':
                self.compile()

    def add(self, layer):

        if not self.layers:
            if layer.config['type'].lower() != 'input':
                raise ValueError('The first layer of the model must be an input layer')
        self.layers.append(layer)
        if layer.__class__ is not list:
            if layer.config['type'] is 'output':
                self.compile()

    def pop(self, loc=-1):

        if len(self.layers) > 0:
            self.layers.pop(loc)

    def switch(self, loc1, loc2):
        self.layers[loc1], self.layers[loc2] = self.layers[loc2], self.layers[loc1]

    def compile(self):
        layers = []
        for layer in self.layers:
            if layer.__class__ is not list:
                layer = [layer]
            layers.append(layer)

        if self.layers[0].config['type'] != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if self.layers[-1].config['type'] != 'output':
            raise ValueError('The last layer of the model must be an output layer')
        s = self.conn
        s.buildmodel(model=dict(name=self.model_name, replace=True), type='RNN')

        layer_num = 1
        block_num = 0

        for layer_s in layers:
            for layer in layer_s:
                if layer.config['type'] == 'input':
                    s.addLayer(model=self.model_name, name='Data',
                               layer=layer.config)
                    layer.name = 'Data'

                else:

                    if layer.config['type'].lower() == 'recurrent':
                        layer_name = 'Recurrent{}_{}'.format(block_num, layer_num)
                        layer_num += 1

                    elif layer.config['type'].lower() == 'output':
                        layer_name = 'Output'

                    else:
                        raise ValueError('{} is not a supported layer type'.format(layer['type']))

                    src_layer_names = [src_layer.name for src_layer in src_layers]

                    s.addLayer(model=self.model_name, name=layer_name,
                               layer=layer.config, srcLayers=src_layer_names)
                    layer.name = layer_name
                    layer.src_layers = src_layers

                print(layer.name)
                print(layer.config)
            layer_num = 1
            block_num += 1
            src_layers = layer_s
