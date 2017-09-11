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

        elif layer.config['type'].lower() == 'batchnorm':
            print('NOTE: Batch Normalization layer added.')

        elif layer.config['type'].lower() == 'block':
            print('NOTE: A block of layers added.')

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
        bn_num = 1
        block_num = 1

        compiled_layers = []

        for layer in self.layers:
            if layer.config['type'] == 'block':
                options = layer.compile(src_layer=output_layer, block_num=block_num)
                block_num += 1
                for item in layer.layers:
                    compiled_layers.append(item)
                output_layer = layer.layers[-1]
                for option in options:
                    conn.retrieve('addlayer', model=self.model_name, **option)
            else:
                if layer.config['type'] == 'input':
                    layer.name = 'Data'
                else:
                    if layer.config['type'].lower() in ('convo', 'convolution'):
                        layer.name = 'Conv{}_{}'.format(block_num, conv_num)
                        conv_num += 1
                    elif layer.config['type'].lower() == 'batchnorm':
                        layer.name = 'BN{}_{}'.format(block_num, bn_num)
                        bn_num += 1
                    elif layer.config['type'].lower() in ('pool', 'pooling'):
                        layer.name = 'Pool{}'.format(block_num)
                        block_num += 1
                        conv_num = 1
                    elif layer.config['type'].lower() in ('fc', 'fullconnect'):
                        layer.name = 'FC{}'.format(fc_num)
                        fc_num += 1
                    elif layer.config['type'].lower() == 'output':
                        layer.name = 'Output'
                    else:
                        raise ValueError('{} is not a supported layer type'.format(layer['type']))
                    layer.src_layers = [output_layer]

                option = layer.to_model_params()
                compiled_layers.append(layer)
                output_layer = layer

                conn.retrieve('addlayer', model=self.model_name, **option)

        print('NOTE: Model compiled successfully.')
        self.layers = compiled_layers
        for layer in self.layers:
            layer.summary()

    def count_params(self):
        count = 0
        for layer in self.layers:

            if layer.num_weights is None:
                num_weights = 0
            else:
                num_weights = layer.num_weights

            if layer.num_bias is None:
                num_bias = 0
            else:
                num_bias = layer.num_bias

            count += num_weights + num_bias
        return int(count)

    def summary(self):
        bar_line = '*' + '=' * 18 + '*' + '=' * 15 + '*' + '=' * 8 + '*' + \
                   '=' * 12 + '*' + '=' * 17 + '*' + '=' * 22 + '*\n'
        h_line = '*' + '-' * 18 + '*' + '-' * 15 + '*' + '-' * 8 + '*' + \
                 '-' * 12 + '*' + '-' * 17 + '*' + '-' * 22 + '*\n'
        title_line = '|{:^18}'.format('Layer (Type)') + \
                     '|{:^15}'.format('Kernel Size') + \
                     '|{:^8}'.format('Stride') + \
                     '|{:^12}'.format('Activation') + \
                     '|{:^17}'.format('Output Size') + \
                     '|{:^22}|\n'.format('Number of Parameters')
        output = bar_line + title_line + h_line
        for layer in self.layers:
            output = output + layer.summary_str
        output = output + bar_line

        output = output + '|Total Number of Parameters: {:<69}|\n'. \
            format(format(self.count_params(), ','))
        output = output + '*' + '=' * 97 + '*'

        self.summary_str = output
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

    keys = ['<Act>Activation:', '<Kernel>Kernel Size:']

    content_dict = dict()

    if layer.kernel_size is not None:
        content_dict['<Kernel>Kernel Size:'] = layer.kernel_size

    if layer.activation is not None:
        if 'act' in layer.config:
            content_dict['<Act>Activation:'] = layer.activation
        if 'pool' in layer.config:
            content_dict['<Act>Pooling:'] = layer.activation

    if layer.type_name is not 'Input':
        title_col = '<Output>Output Size:}|'
        value_col = '{}'.format(layer.output_size) + '}'

        for key in keys:
            if key in content_dict.keys():
                title_col = key + '|' + title_col
                value_col = '{}'.format(content_dict[key]) + '|' + value_col
    else:
        title_col = '<Output>Input Size:}|'
        value_col = '{}'.format(layer.output_size) + '}'

    label = cell1 + '|{' + title_col + '{' + value_col
    label = r'{}'.format(label)

    return dict(name=layer.name, label=label, fillcolor=layer.color_code)


def layer_to_edge(layer):
    gv_params = []
    for item in layer.src_layers:
        gv_params.append(dict(tail_name='{}'.format(item.name),
                              head_name='{}'.format(layer.name),
                              len='0.2'))
    return gv_params


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
            for gv_param in layer_to_edge(layer):
                model_graph.edge(**gv_param)

    return model_graph
