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

''' Utility functions for the DLPy package '''

import os
import random
import re
import string

import matplotlib.pyplot as plt
import numpy as np
import six
import swat as sw
from swat.cas.table import CASTable


def random_name(name='ImageData', length=6):
    '''
    Generate random name

    Parameters
    ----------
    name : string, optional
        Prefix of the generated name
    length : int, optional
        Length of the random characters in the name

    Returns
    -------
    string

    '''
    return name + '_' + ''.join(random.sample(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, length))


def input_table_check(input_table):
    '''
    Unify the input_table format

    Parameters
    ----------
    input_table : CASTable or string or dict
        Input table specification

    Returns
    -------
    dict
        Input table parameters

    '''
    if isinstance(input_table, six.string_types):
        input_table = dict(name=input_table)
    elif isinstance(input_table, dict):
        input_table = input_table
    elif isinstance(input_table, CASTable):
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
    Compute the product of an iterable array with None as its element

    Parameters
    ----------
    array : iterable-of-numeric
        The numbers to use as input

    Returns
    -------
    numeric
        Product of all the elements of the array

    '''
    prod = 1
    for i in array:
        if i is not None:
            prod *= i
    return prod


def get_max_size(start_path='.'):
    '''
    Get the max size of files in a folder including sub-folders

    Parameters
    ----------
    start_path : string, optional
        The directory to start the file search

    Returns
    -------
    int

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
    Determine blocksize according to imagesize in the table

    Parameters
    ----------
    width : int
        The width of the image
    height : int
        The height of the image

    Returns
    -------
    int

    '''
    return width * height * 3 / 1024


def predicted_prob_barplot(ax, labels, values):
    '''
    Generate a horizontal barplot for the predict probability

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot to
    labels : list-of-strings
        Predicted class labels
    values : list-of-numeric
        Predicted probabilities

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    '''
    y_pos = (0.2 + np.arange(len(labels))) / (1 + len(labels))
    width = 0.8 / (1 + len(labels))
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    for i in range(len(labels)):
        ax.barh(y_pos[i], values[i], width, align='center',
                color=colors[i], ecolor='black')
        ax.text(values[i] + 0.01, y_pos[i], '{:.2%}'.format(values[i]))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, rotation=45)
    ax.set_xlabel('Probability')
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title('Predicted Probability')
    return ax


def plot_predict_res(image, label, labels, values):
    '''
    Generate a side by side plot of the predicted result

    Parameters
    ----------
    image :
        Specifies the orginal image to be classified.
    label : string
        Specifies the class name of the image.
    labels : list-of-strings
        Predicted class labels
    values : list-of-numeric
        Predicted probabilities

    Returns
    -------
    :class:`matplotlib.axes.Axes`

    '''
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('{}'.format(label))
    ax1.imshow(image)
    ax1.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    predicted_prob_barplot(ax2, labels, values)


def camelcase_to_underscore(strings):
    ''' Convert camelcase to underscore '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', strings)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def add_caslib(conn, path):
    '''
    Add a new caslib, as needed

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    path : string
        Specifies the server-side path to check

    Returns
    -------
    string
        The name of the caslib pointing to the path

    '''
    if path in conn.caslibinfo().CASLibInfo.Path.tolist():
        cas_lib_name = conn.caslibinfo().CASLibInfo[
            conn.caslibinfo().CASLibInfo.Path == path]['Name']

        return cas_lib_name.tolist()[0]
    else:
        cas_lib_name = random_name('Caslib', 6)
        conn.retrieve('table.addcaslib', message_level='error',
                      name=cas_lib_name, path=path, activeOnAdd=False,
                      dataSource=dict(srcType='DNFS'))
        return cas_lib_name


def upload_astore(conn, path, table_name=None):
    '''
    Load the local astore file to server

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    path : string
        Specifies the client-side path of the astore file
    table_name : string, or casout options
        Specifies the name of the cas table on server to put the astore object

    '''
    conn.loadactionset('astore')

    with open(path, 'br') as f:
        astore_byte = f.read()

    store_ = sw.blob(astore_byte)

    if table_name is None:
        table_name = random_name('ASTORE')
    conn.astore.upload(rstore=table_name, store=store_)


def unify_keys(dic):
    '''
    Change all the key names in a dictionary to lower case, remove "_" in the key names.

    Parameters
    ----------
    dic : dict

    Returns
    -------
    dict
        dictionary with updated key names

    '''

    old_names = list(dic.keys())
    new_names = [item.lower().replace('_', '') for item in old_names]
    for new_name, old_name in zip(new_names, old_names):
        dic[new_name] = dic.pop(old_name)

    return dic


def check_caslib(conn, path):
    '''
    Check whether the specified path is in the caslibs of the current session.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object

    path : str
        Specifies the name of the path.

    Returns
    -------
    flag : bool
        Specifies if path exist in session.
    caslib_name : str (if exist)
        Specifies the name of the caslib that contain the path.

    '''
    paths = conn.caslibinfo().CASLibInfo.Path.tolist()
    caslibs = conn.caslibinfo().CASLibInfo.Name.tolist()

    if path in paths:
        caslibname = caslibs[paths.index(path)]
        return True, caslibname
    else:
        return False
