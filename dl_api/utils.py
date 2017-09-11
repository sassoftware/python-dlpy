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
    elif type_indicator in ("ImageTable", "CASTable"):
        input_table = input_table.to_table_params()
    else:
        raise TypeError('input_table must be one of the following:\n'
                        '1. A CAS table object;\n'
                        '2. A string specifies the name of the CAS table,\n'
                        '3. A dictionary specifies the CAS table\n'
                        '4. An ImageTable object.')
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


def image_blocksize(width, height):
    '''
    Function to determine blocksize according to imagesize in the table.
    '''
    return width * height * 3 / 1024


def predicted_prob_barplot(ax, labels, values):
    '''
    Function to generate a horizontal barplot for the predict probability.

    Parameters:

    ----------

    ax : a matplotlib.axes.Axes object.

    labels: class labels.

    values: predicted probabilities.

    Return:

    ----------

    ax : a matplotlib.axes.Axes object including the barplot.


    '''

    import numpy as np

    y_pos = (0.2 + np.arange(len(labels))) / (1 + len(labels))
    width = 0.8 / (1 + len(labels))
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    for i in range(len(labels)):
        ax.barh(y_pos[i], values[i], width, align='center', color=colors[i], ecolor='black')
        ax.text(values[i] + 0.01, y_pos[i], '{:.2%}'.format(values[i]))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, rotation=45)
    ax.set_xlabel('Probability')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.1])
    ax.set_xticklabels(["0%", '25%', '50%', '75%', "100%"])
    ax.set_title('Predicted Probability')

    return ax


def plot_predict_res(image, label, labels, values):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('{}'.format(label))
    ax1.imshow(image)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    predicted_prob_barplot(ax2, labels, values)


def camelcase_to_underscore(strings):
    import re

    '''
    Function to convert camelcase to underscore.
    '''

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strings)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
