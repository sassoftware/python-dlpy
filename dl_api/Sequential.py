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
Sequential object for deep learning.
'''

from .model import Model


class Sequential(Model):
    def __init__(self, sess, layers=None, model_name=None):
        if not sess.queryactionset('deepLearn')['deepLearn']:
            sess.loadactionset('deepLearn')

        Model.__init__(self, sess, model_name=model_name)

        if layers is None:
            self.layers = []
        elif type(layers) is not dict:
            raise TypeError('layers has to be a list of layer(s).')
        else:
            self.layers = layers
            if layers[-1]['type'] == 'output':
                self.compile()

    def add(self, layer):

        if self.layers == [] and layer.config['type'].lower() != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if len(self.layers) > 0 and layer.config['type'] is 'input':
            raise ValueError('Only the first layer of the Sequential model can be an input layer')
        self.layers.append(layer)
        if layer.config['type'] is 'output':
            self.compile()

    def pop(self, loc=-1):

        if len(self.layers) > 0:
            self.layers.pop(loc)

    def switch(self, loc1, loc2):
        self.layers[loc1], self.layers[loc2] = self.layers[loc2], self.layers[loc1]

    def compile(self):
        if self.layers[0].config['type'] != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if self.layers[-1].config['type'] != 'output':
            raise ValueError('The last layer of the model must be an output layer')
        s = self.sess
        s.buildmodel(model=dict(name=self.model_name, replace=True), type='CNN')

        conv_num = 1
        fc_num = 1
        block_num = 1

        for layer in self.layers:
            if layer.config['type'] == 'input':
                s.addLayer(model=self.model_name, name='Data',
                           layer=layer.config)
                layer.name = 'Data'

            else:

                if layer.config['type'].lower() in ('convo', 'convolution'):
                    layer_name = 'Conv{}_{}'.format(block_num, conv_num)
                    conv_num += 1

                elif layer.config['type'].lower() in ('pool', 'pooling'):
                    layer_name = 'Pool{}'.format(block_num)
                    block_num += 1
                    conv_num = 1

                elif layer.config['type'].lower() in ('fc', 'fullconnect'):
                    layer_name = 'FC{}'.format(fc_num)
                    fc_num += 1

                elif layer.config['type'].lower() == 'output':
                    layer_name = 'Output'

                else:
                    raise ValueError('{} is not a supported layer type'.format(layer['type']))

                print(layer_name)
                print(layer.config)
                s.addLayer(model=self.model_name, name=layer_name,
                           layer=layer.config, srcLayers=src_layers.name)
                layer.name = layer_name
                layer.src_layers = src_layers
            src_layers = layer

    def summary(self):
        bar_line = '*' + '=' * 15 + '*' + '=' * 15 + '*' + '=' * 8 + '*' + \
                   '=' * 12 + '*' + '=' * 17 + '*' + '=' * 22 + '*\n'
        h_line = '*' + '-' * 15 + '*' + '-' * 15 + '*' + '-' * 8 + '*' + \
                 '-' * 12 + '*' + '-' * 17 + '*' + '-' * 22 + '*\n'
        title_line = '|{:^15}'.format('Layer (Type)') + \
                     '|{:^15}'.format('Kernel Size') + \
                     '|{:^8}'.format('Stride') + \
                     '|{:^12}'.format('Activation') + \
                     '|{:^17}'.format('Output Size') + \
                     '|{:^22}|\n'.format('Number of Parameters')
        output = bar_line + title_line + h_line
        for layer in self.layers:
            output = output + layer.summary()
        output = output + bar_line
        print(output)
