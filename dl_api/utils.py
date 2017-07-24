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

import random
import string
import os


def random_name(name='ImageData', length=6):
    '''
    Function to generate ramdom name.
    '''

    return name + '_' + ''.join(random.sample(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, length))


def input_table_check(input_table):
    '''
    Function to unify the input_table format.


    Parameters:

    ----------

    input_table : A CAS table object, a string specifies the name of the CAS table,
                a dictionary specifies the CAS table, or an Image object.

    Return:

    ----------
    A dictionary specifies the CAS table

    '''

    type_indicator = input_table.__class__.__name__
    if type_indicator == "str":
        input_table = dict(name=input_table)
    elif type_indicator == "dict":
        input_table = input_table
    elif type_indicator == "Image":
        input_table = input_table.tbl
    elif type_indicator == "CASTable":
        input_table = dict(name=input_table.tableinfo().TableInfo.Name[0])
    else:
        raise TypeError('input_table must be one of the following:\n'
                        '1. A CAS table object;\n'
                        '2. A string specifies the name of the CAS table,\n'
                        '3. A dictionary specifies the CAS table\n'
                        '4. An Image object.')
    return input_table


def prod_without_none(array):
    '''
    Function to compute the product of an iterable array with None as its element.


    Parameters:

    ----------

    array : an iterable array, e.g. list, tuple, numpy array.

    Return:

    ----------
    prod : the product of all the elements of the array.

    '''
    prod = 1
    for i in array:
        if i is not None:
            prod *= i
    return prod


def get_max_size(start_path='.'):
    '''
    Function to get the max size of files in a folder including sub-folders.
    '''
    max_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            file_size = os.path.getsize(fp)
            if file_size > max_size:
                max_size = file_size
    return max_size


def update_blocksize(width, height):
    '''
    Function to determine blocksize according to imagesize in the table.
    '''
    return width * height * 3 * 8 / 1024


def layer_to_node(layer):
    cell1 = r'{}\n({})'.format(layer.name, layer.config['type'])
    cell21 = '<Kernel> Kernel Size:'
    cell22 = '<Output> Output Size:'
    cell31 = '{}'.format(layer.kernel_size)
    cell32 = '{}'.format(layer.output_size)

    label = cell1 + '|{' + cell21 + '|' + cell22 + '}|' + '{' + cell31 + '|' + cell32 + '}'
    label = r'{}'.format(label)
    return dict(name=layer.name, label=label, fillcolor=layer._color_code_)


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
