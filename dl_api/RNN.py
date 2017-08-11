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

        if layer.__class__ is not list:
            layer = [layer]

        if not self.layers:
            if layer[0].config['type'].lower() != 'input':
                raise ValueError('The first layer of the model must be an input layer')
        for _layer in layer:
            if _layer.config['type'].lower() == 'input':
                print('NOTE: An input layer is add to the model.')

            # elif _layer.config['type'].lower() in ('convo', 'convolution'):
            #     print('NOTE: A convolutional layer is add to the model.')
            #
            # elif _layer.config['type'].lower() in ('pool', 'pooling'):
            #     print('NOTE: A pooling layer is add to the model.')
            #
            # elif _layer.config['type'].lower() in ('fc', 'fullconnect'):
            #     print('NOTE: A fully-connected layer is add to the model.')

            elif _layer.config['type'].lower() in ('rnn', 'recurrent'):
                print('NOTE: Recurrent layer added.')

            elif _layer.config['type'].lower() == 'output':
                print('NOTE: Output layer added')

        self.layers.append(layer)
        if layer[0].config['type'] is 'output':
            self.compile()

    def pop(self, loc=-1):
        if len(self.layers) > 0:
            self.layers.pop(loc)

    def switch(self, loc1, loc2):
        self.layers[loc1], self.layers[loc2] = self.layers[loc2], self.layers[loc1]

    def compile(self):

        if self.layers[0][0].config['type'] != 'input':
            raise ValueError('The first layer of the model must be an input layer')
        if self.layers[-1][0].config['type'] != 'output':
            raise ValueError('The last layer of the model must be an output layer')
        conn = self.conn
        conn.retrieve('buildmodel', model=dict(name=self.model_name, replace=True), type='RNN')

        layer_num = 1
        block_num = 0

        for layer_s in self.layers:
            for layer in layer_s:
                if layer.config['type'] == 'input':
                    conn.retrieve('addlayer', model=self.model_name, name='Data',
                                  layer=layer.config)
                    layer.name = 'Data'

                else:

                    if layer.config['type'].lower() == 'recurrent':
                        layer_name = 'Rec_{}_{}'.format(block_num, layer_num)
                        layer_num += 1

                    elif layer.config['type'].lower() == 'output':
                        layer_name = 'Output'

                    else:
                        raise ValueError('{} is not a supported layer type'.format(layer['type']))

                    src_layer_names = [src_layer.name for src_layer in src_layers]

                    conn.retrieve('addlayer', model=self.model_name, name=layer_name,
                                  layer=layer.config, srcLayers=src_layer_names)
                    layer.name = layer_name
                    layer.src_layers = src_layers

            layer_num = 1
            block_num += 1
            src_layers = layer_s

        print('NOTE: Model compiled successfully.')

    def summary(self):
        bar_line = '*' + '=' * 20 + '*' + '=' * 14 + '*' + '=' * 10 + '*' + \
                   '=' * 20 + '*' + '=' * 12 + '*' + '=' * 8 + '*' + '=' * 17 + '*\n'
        h_line = '*' + '-' * 20 + '*' + '-' * 14 + '*' + '-' * 10 + '*' + \
                 '-' * 20 + '*' + '-' * 12 + '*' + '-' * 8 + '*' + '-' * 17 + '*\n'
        title_line = '|{:^20}'.format('Layer (Type)') + \
                     '|{:^14}'.format('Recurrent Type') + \
                     '|{:^10}'.format('Activation') + \
                     '|{:^20}'.format('Source Layer(s)') + \
                     '|{:^12}'.format('Output Type') + \
                     '|{:^8}'.format('Reversed') + \
                     '|{:^17}|\n'.format('Number of Neurons')
        output = bar_line + title_line + h_line
        for layer_s in self.layers:
            for layer in layer_s:
                name = '{}({})'.format(layer.name, layer.config['type'])
                col1 = '| {:<19}'.format('{}'.format(name))

                if 'rnnType' not in layer.config.keys():
                    col2 = '|{:^14}'.format('N/A')
                else:
                    col2 = '|{:^14}'.format('{}'.format(layer.config['rnnType']))

                if 'act' not in layer.config.keys():
                    col3 = '|{:^10}'.format('N/A')
                else:
                    col3 = '|{:^10}'.format('{}'.format(layer.config['act']))

                if layer.src_layers is None:
                    col4 = '|{:^20}'.format('N/A')
                else:
                    src_layer_names = [src_layer.name for src_layer in layer.src_layers]
                    col4 = '|{:^20}'.format(', '.join(src_layer_names))

                if 'outputType' not in layer.config.keys():
                    col5 = '|{:^12}'.format('N/A')
                else:
                    col5 = '|{:^12}'.format('{}'.format(layer.config['outputType']))

                if 'Reversed' not in layer.config.keys():
                    col6 = '|{:^8}'.format('N/A')
                else:
                    col6 = '|{:^8}'.format('{}'.format(layer.config['Reversed']))

                if 'n' not in layer.config.keys():
                    col7 = '|{:^17}|\n'.format('N/A')
                else:
                    col7 = '|{:^17}|\n'.format('{}'.format(layer.config['n']))

                layer_summary = col1 + col2 + col3 + col4 + col5 + col6 + col7
                output = output + layer_summary
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
    cell21 = '<Type> Type:'
    cell22 = '<Neuron> Neuron:'

    if 'rnnType' not in layer.config.keys():
        cell31 = 'N/A'
    else:
        cell31 = '{}'.format(layer.config['rnnType'])

    if 'n' not in layer.config.keys():
        cell32 = 'N/A'
    else:
        cell32 = '{}'.format(layer.config['n'])

    label = cell1 + '|{' + cell21 + '|' + cell22 + '}|' + '{' + cell31 + '|' + cell32 + '}'
    label = r'{}'.format(label)
    return dict(name=layer.name, label=label, fillcolor=layer.color_code)


def layer_to_edge(layer):
    options = []
    for src_layer in layer.src_layers:
        options.append(dict(tail_name='{}'.format(src_layer.name),
                            head_name='{}'.format(layer.name),
                            len='0.2'))
    return options


def model_to_graph(model):
    import graphviz as gv
    model_graph = gv.Digraph(name=model.model_name,
                             node_attr=dict(shape='record', style='filled,rounded'))
    # can be added later for adjusting figure size.
    # fixedsize='True', width = '4', height = '1'))

    model_graph.attr(label=r'DAG for {}:'.format(model.model_name),
                     labelloc='top', labeljust='left')
    model_graph.attr(fontsize='20')

    for layer_s in model.layers:
        for layer in layer_s:
            if layer.config['type'].lower() == 'input':
                model_graph.node(**layer_to_node(layer))
            else:
                model_graph.node(**layer_to_node(layer))
                for option in layer_to_edge(layer):
                    model_graph.edge(**option)

    return model_graph
