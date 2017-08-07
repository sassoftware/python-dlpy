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

        if self.layers == [] and layer.config['type'].lower() != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if len(self.layers) > 0 and layer.config['type'] is 'input':
            raise ValueError('Only the first layer of the Sequential model can be an input layer')
        self.layers.append(layer)

        if layer.config['type'].lower() == 'input':
            print('NOTE: Input layer added.')

        elif layer.config['type'].lower() in ('convo', 'convolution'):
            print('NOTE: Convolutional layer added.')

        elif layer.config['type'].lower() in ('pool', 'pooling'):
            print('NOTE: Pooling layer added.')

        elif layer.config['type'].lower() in ('fc', 'fullconnect'):
            print('NOTE: Fully-connected layer added.')

        elif layer.config['type'].lower() == 'output':
            print('NOTE: Output layer added.')
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
        conn = self.conn
        conn.retrieve('buildmodel', model=dict(name=self.model_name, replace=True), type='CNN')

        conv_num = 1
        fc_num = 1
        block_num = 1

        for layer in self.layers:
            if layer.config['type'] == 'input':
                conn.retrieve('addlayer', model=self.model_name, name='Data',
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

                conn.retrieve('addlayer', model=self.model_name, name=layer_name,
                              layer=layer.config, srcLayers=src_layers.name)
                layer.name = layer_name
                layer.src_layers = src_layers

            src_layers = layer
        print('NOTE: Model compiled successfully.')

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

    def plot_network(self):
        '''
        Function to plot the model DAG
        '''

        from IPython.display import display
        import os
        os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

        display(model_to_graph(self))


def layer_to_node(layer):
    cell1 = r'{}\n({})'.format(layer.name, layer.config['type'])
    cell21 = '<Kernel> Kernel Size:'
    cell22 = '<Output> Output Size:'
    cell31 = '{}'.format(layer.kernel_size)
    cell32 = '{}'.format(layer.output_size)

    label = cell1 + '|{' + cell21 + '|' + cell22 + '}|' + '{' + cell31 + '|' + cell32 + '}'
    label = r'{}'.format(label)
    return dict(name=layer.name, label=label, fillcolor=layer.color_code)


def layer_to_edge(layer):
    return dict(tail_name='{}'.format(layer.src_layers.name),
                head_name='{}'.format(layer.name),
                len='0.2')


def model_to_graph(model):
    import graphviz as gv
    model_graph = gv.Digraph(name=model.model_name,
                             node_attr=dict(shape='record', style='filled,rounded'))
    # can be added later for adjusting figure size.
    # fixedsize='True', width = '4', height = '1'))

    model_graph.attr(label=r'DAG for {}:'.format(model.model_name),
                     labelloc='top', labeljust='left')
    model_graph.attr(fontsize='20')

    for layer in model.layers:
        if layer.config['type'].lower() == 'input':
            model_graph.node(**layer_to_node(layer))
        else:
            model_graph.node(**layer_to_node(layer))
            model_graph.edge(**layer_to_edge(layer))

    return model_graph
