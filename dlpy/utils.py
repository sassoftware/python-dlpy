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

import json
import os
import platform
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import re
import six
import swat as sw
import string
import xml.etree.ElementTree as ET
from swat.cas.table import CASTable
from PIL import Image
import warnings
import platform
import collections
from itertools import repeat
import math


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


def multiply_elements(parameter_counts):
    '''
    Compute the product of an iterable array with None as its element

    Parameters
    ----------
    array : iterable-of-numeric
        The numbers to use as input

    Returns
    -------
    number
        Product of all the elements of the array

    '''

    result = 1
    for i in parameter_counts:
        if i is not None:
            result *= i
    return result


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


def underscore_to_camelcase(strings):
    ''' Convert underscore to camelcase '''
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), strings)


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


def find_caslib(conn, path):
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
    string
        Specifies the name of the caslib that contains the path.

    '''
    paths = conn.caslibinfo().CASLibInfo.Path.tolist()
    caslibs = conn.caslibinfo().CASLibInfo.Name.tolist()

    server_type = get_cas_host_type(conn).lower()

    if server_type.startswith("lin") or server_type.startswith("osx"):
        sep = '/'
    else:
        sep = '\\'

    if not path.endswith(sep):
        path += sep

    if path in paths:
        caslibname = caslibs[paths.index(path)]
        return caslibname
    else:
        return None


def get_imagenet_labels_table(conn, label_length=None):
    temp_name = random_name('new_label_table', 6)

    filename = os.path.join('datasources', 'imagenet_labels.csv')
    project_path = os.path.dirname(os.path.abspath(__file__))
    full_filename = os.path.join(project_path, filename)

    return get_user_defined_labels_table(conn, full_filename, label_length)


def get_user_defined_labels_table(conn, label_file_name, label_length=None):
    temp_name = random_name('new_label_table', 6)

    full_filename = label_file_name

    if label_length is None:
        char_length = 200
    else:
        char_length = label_length
    
    labels = pd.read_csv(full_filename, skipinitialspace=True, index_col=False)
    conn.upload_frame(labels, casout=dict(name=temp_name, replace=True),
                      importoptions={'vars':[
                          {'name': 'label_id', 'type': 'int64'},
                          {'name': 'label', 'type': 'char', 'length': char_length}]})

    return conn.CASTable(temp_name)

    
def get_server_path_sep(conn):
    '''
    Get the directory separator of server.

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection

    Returns
    -------
    string
        Directory separator

    '''
    server_type = get_cas_host_type(conn).lower()
    sep = '\\'
    if server_type.startswith("lin") or server_type.startswith("osx"):
        sep = '/'
    return sep


def caslibify(conn, path, task='save'):
    '''
    This is a utility function to find or create a caslib for a given path and for a given task.
    This function also checks the root folders to see if there is a caslib created, if so return that caslib along
    with the normalized path. Otherwise creates one.

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection
    path : string
        Specifies the path to be analyzed for creating a caslib.
    task : string
        Specifies the task. If it is a load task, then a caslib needs to be created to the parent folder
        of the path folder. If it is a save task, then a caslib needs to be created to the path folder.
    '''
    if task == 'save':

        sep = get_server_path_sep(conn)

        if path.endswith(sep):
            path = path[:-1]

        path_split = path.split(sep)
        caslib = None
        new_path = sep
        if path.startswith(sep):
            start = 1
        else:
            start = 0

        end = len(path_split)
        while caslib is None and start < end:

            new_path += path_split[start]+sep
            caslib = find_caslib(conn, new_path)
            start += 1

        remaining_path = ''
        for i in range(start, end):
            remaining_path += path_split[i]
            remaining_path += sep

        if caslib is not None:
            return caslib, remaining_path, False
        else:
            new_caslib = random_name('Caslib', 6)
            rt = conn.retrieve('addcaslib', _messagelevel='error', name=new_caslib, path=path,
                               activeonadd=False, subdirectories=True, datasource={'srctype': 'path'})

            if rt.severity > 1:
                raise DLPyError('something went wrong while adding the caslib for the specified path.')
            else:
                return new_caslib, '', True
    else:
        server_type = get_cas_host_type(conn).lower()
        if server_type.startswith("lin") or server_type.startswith("osx"):
            path_split = path.rsplit("/", 1)
        else:
            path_split = path.rsplit("\\", 1)

        if len(path_split) == 2:
            caslib = find_caslib(conn, path_split[0])
            if caslib is not None:
                return caslib, path_split[1], False
            else:
                new_caslib = random_name('Caslib', 6)
                rt = conn.retrieve('addcaslib', _messagelevel='error', name=new_caslib, path=path_split[0],
                                   activeonadd=False, subdirectories=True, datasource={'srctype': 'path'})

                if rt.severity > 1:
                    print('Something went wrong. Most likely, one of the subpaths of the provided path'
                          'is part of an existing caslib. A workaround is to put the file under that subpath or'
                          'move to a different location. It sounds and is inconvenient but it is to protect '
                          'your privacy granted by your system admin.')
                    return None, None, False
                else:
                    return new_caslib, path_split[1], True
        else:
            raise DLPyError('we need more than one level of directories. e.g., /dir1/dir2 ')


def find_path_of_caslib(conn, caslib):
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
    string
        Specifies the name of the caslib that contains the path

    '''
    paths = conn.caslibinfo().CASLibInfo.Path.tolist()
    caslibs = conn.caslibinfo().CASLibInfo.Name.tolist()

    if caslib in caslibs:
        return paths[caslibs.index(caslib)]
    else:
        return None


def get_cas_host_type(conn):
    ''' Return a server type indicator '''
    with sw.option_context(print_messages = False):
        out = conn.about()
    ostype = out['About']['System']['OS Family']
    stype = 'mpp'
    htype = 'nohdfs'
    if out['server'].loc[0, 'nodes'] == 1:
        stype = 'smp'
    if ostype.startswith('LIN'):
        ostype = 'linux'
    elif ostype.startswith('WIN'):
        ostype = 'windows'
    elif ostype.startswith('OSX'):
        ostype = 'mac'
    else:
        raise ValueError('Unknown OS type: ' + ostype)

    # Check to see if HDFS is present
    out = conn.table.querycaslib(caslib='CASUSERHDFS')
    for key, value in list(out.items()):
        if 'CASUSERHDFS' in key and value:
            # Default HDFS caslib for user exists
            htype = ''

    if stype == 'mpp' and (len(htype) > 0):
        return ostype + '.' + stype + '.' + htype
    else:
        return ostype + '.' + stype


class DLPyError(Exception):
    pass


class Box(object):
    '''
    Box class used in object detection

    Parameters
    ----------
    x : float
        x location of box between 0 and 1 relative to image column position
    y : float
        y location of box between 0 and 1 relative to image row position
    w : float
        width of the box
    h : float
        height of the box

    Attributes
    ----------
    x : float
        x location of box between 0 and 1 relative to image column position
    y : float
        y location of box between 0 and 1 relative to image row position
    w : float
        width of the box between 0 and 1 relative to image width
    h : float
        height of the box between 0 and 1 relative to image height

    '''
    def __init__(self, x, y, w, h, class_type = None, confidence = 1.0, image_name = None, format_type = 'xywh'):
        if format_type == 'xywh':
            self.x = x
            self.y = y
            self.w = w
            self.h = h
            self.x_min = x - (w / 2)
            self.x_max = x + (w / 2)
            self.y_min = y - (h / 2)
            self.y_max = y + (h / 2)
        elif 'xyxy':
            self.x_min = x
            self.x_max = y
            self.y_min = w
            self.y_max = h
        self.class_type = class_type
        self.confidence = confidence
        self.image_name = image_name

    def __repr__(self):
        return repr((self.x_min, self.x_max, self.y_min, self.y_max, self.class_type, self.confidence, self.image_name))

    def get_area(self):
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @classmethod
    def iou(cls, box_a, box_b):
        if not isinstance(box_a, Box):
            raise DLPyError('box_a should be Box class')
        if not isinstance(box_b, Box):
            raise DLPyError('box_b should be Box class')
        intersect_area = Box.boxes_intersection(box_a, box_b)
        union_area = box_a.get_area() + box_b.get_area() - intersect_area
        return intersect_area / union_area

    @classmethod
    def boxes_intersection(cls, box_a, box_b):
        if box_a.x_min > box_b.x_max:
            return 0
        if box_b.x_min > box_a.x_max:
            return 0
        if box_a.y_min > box_b.y_max:
            return 0
        if box_b.y_min > box_a.y_max:
            return 0
        x_intersect_left = max(box_a.x_min, box_b.x_min)
        x_intersect_right = min(box_a.x_max, box_b.x_max)
        y_intersect_top = max(box_a.y_min, box_b.y_min)
        y_intersect_bottom = min(box_a.y_max, box_b.y_max)
        return (x_intersect_right - x_intersect_left) * (y_intersect_bottom - y_intersect_top)


def get_iou_distance(box, centroid):
    '''
    Gets one minus the area intersection over union of two boxes

    Parameters
    ----------
    box : Box
        A Box object containing width and height information
    centroid : Box
        A Box object containing width and height information

    Returns
    -------
    float
        One minus intersection over union
        Smaller the number the closer the boxes

    '''
    w = min(box.w, centroid.w)
    h = min(box.h, centroid.h)
    intersection = w*h
    union = box.w * box.h + centroid.w * centroid.h - intersection
    iou = intersection/union
    return 1-iou


def run_kmeans(boxes, centroids, n_anchors=5):
    '''
    Runs a single iteration of the k-means algorithm

    Parameters
    ----------
    boxes : list
        Contains Box objects with width and height labels from training data
    centroids : list
        List of boxes containing current width and height info
        for each cluster cluster center
    n_anchors : int
        Number of anchors for each grid cell

    Returns
    -------
    ( list, list, float)
        new_centroids : updated list of Box objects containing width and height
        of each anchor box
        clusters : List of list of Box objects grouped by which centroid they
        are closest to.
        loss : sum of distances of each Box in training data to its closest centroid

    '''
    loss = 0
    clusters = []
    new_centroids = []
    for i in range(n_anchors):
        clusters.append([])
        new_centroids.append(Box(0, 0, 0, 0))
    for box in boxes:
        min_distance = 1
        group_index = 0
        for centroid_index, centroid in enumerate(centroids):
            distance = get_iou_distance(box, centroid)
            if distance < min_distance:
                min_distance = distance
                group_index = centroid_index
        clusters[group_index].append(box)
        loss += min_distance
        new_centroids[group_index].w += box.w
        new_centroids[group_index].h += box.h

    for i in range(n_anchors):
        if len(clusters[i]) == 0:
            random_choice = random.choice(boxes)
            new_centroids[i].w = random_choice.w
            new_centroids[i].h = random_choice.h
            clusters[i].append(random_choice)
        new_centroids[i].w /= len(clusters[i])
        new_centroids[i].h /= len(clusters[i])

    return new_centroids, clusters, loss


def get_anchors(conn, data, coord_type, image_size=None, grid_number=13,
                n_anchors=5, loss_convergence=1e-5):
    '''
    Gets best n_anchors for object detection grid cells based on k-means.

    Parameters
    ----------
    conn : session
        Specifies the session.
    data : CASTable
        Specifies the table containing object detection box labels
    coord_type : string
        Specifies the format of the box labels
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
        'coco' specifies xmin, ymin, xmax, ymax that are borders of a bounding boxes.
        The values are relative to parameter image_size.
        Valid Values: yolo, coco
    image_size : int, optional
        Specifies the size of images.
        The parameter image_size is optional if coord_type = 'yolo'
        Default: None
    grid_number : int, optional
        The number of grids in each row or column.
        Total number of grids = grid_number*grid_number
        Default: 13
    n_anchors : int, optional
        Number of anchors in each grid cell
        Default: 5
    loss_convergence : float, optional
        If the change in k-means loss from one iteration to the next
        is smaller than loss_convergence then k-means has converged
        Default: 1e-5

    Returns
    -------
    tuple
        Contains widths and heights of anchor boxes

    '''
    if coord_type.lower() == 'coco' and image_size is None:
        raise ValueError('image_size must be specified if coord_type is coco format.')
    boxes = []
    keep_cols = []
    input_tbl_opts = input_table_check(data)
    anchor_tbl = conn.CASTable(**input_tbl_opts)
    for col_name in anchor_tbl.columns:
        if any(s in col_name for s in ['width', 'height', '_nObjects_', 'min', 'max']):
            keep_cols.append(col_name)
    anchor_tbl = anchor_tbl[keep_cols]
    anchor_tbl = anchor_tbl[~anchor_tbl['_nObjects_'].isNull()]

    for idx, row in anchor_tbl.iterrows():
        n_object = int(row.loc['_nObjects_'])
        for i in range(n_object):
            if coord_type.lower() == 'yolo':
                width = float(row.loc['_Object{}_width'.format(i)])
                height = float(row.loc['_Object{}_height'.format(i)])
            elif coord_type.lower() == 'coco':
                width = float((row.loc['_Object{}_xmax'.format(i)] - row.loc['_Object{}_xmin'.format(i)]) / image_size)
                height = float((row.loc['_Object{}_ymax'.format(i)] - row.loc['_Object{}_ymin'.format(i)]) / image_size)
            else:
                print('Error: Only support Yolo and CoCo coordType so far')
                return
            boxes.append(Box(0, 0, width, height))
    centroid_indices = np.random.choice(len(boxes), n_anchors)
    centroids = []
    for centroid_index in centroid_indices:
        centroids.append(boxes[centroid_index])

    centroids, clusters, old_loss = run_kmeans(boxes, centroids, n_anchors)

    while True:
        centroids, clusters, loss = run_kmeans(boxes, centroids, n_anchors)
        if abs(old_loss - loss) < loss_convergence:
            break
        old_loss = loss
    anchors = []
    for centroid in centroids:
        anchors += [centroid.w * grid_number, centroid.h * grid_number]
    return tuple(anchors)


def get_max_objects(cas_table):
    '''
    Get the maximum number of objects in an image from all instances in dataset

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table containing the object detection data.
        Must contain column '_nObjects_'

    Returns
    -------
    int
        Maximum number of objects found in an image in the dataset

    '''
    if isinstance(cas_table, CASTable):
        pass
    else:
        raise ValueError('Input table not valid name or CASTable')
    if '_nObjects_' not in cas_table.columns.tolist():
        raise ValueError('Input table must contain _nObjects_ column')
    if (cas_table['_nObjects_'] < 1).all():
        raise ValueError('CASTable {} contains no images with labeled objects'.format(cas_table.name))

    summary = cas_table.summary()['Summary']
    return int(summary[summary['Column'] == '_nObjects_']['Max'].tolist()[0])


def filter_by_filename(cas_table, filename, filtered_name=None):
    '''
    Filters CASTable by filename using '_path_' or '_filename_0' column

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table to be filtered
        Note: CASTable should have a '_path_' or '_filename_0' column
        and an '_id_' column
    filename : string
        Can be part or full name of image or path
        If not unique, returns all that contain filename
    filtered_name : string, optional
        Name of output table

    Returns
    -------
    CASTable
        Filtered table containing all instances that have

    '''
    if filtered_name:
        if isinstance(filtered_name, str):
            new_name = filtered_name
        else:
            raise ValueError('filtered_name must be a str or leave as None to generate a random name')
    else:
        new_name = random_name('filtered')

    if '_path_' not in cas_table.columns.tolist() and '_filename_0' not in cas_table.columns.tolist():
        raise ValueError('\'_path_\' or \'_filename_0\' column not found in CASTable : {}'.format(cas_table.name))
    if isinstance(filename, list):
        if not all(isinstance(x, str) for x in filename):
            raise ValueError('filename must be a string or a list of strings')
        image_id = []
        for name in filename:
            if '_path_' in cas_table.columns.tolist():
                id_num = cas_table[cas_table['_path_'].str.contains(name)]['_id_'].tolist()
                if id_num:
                    image_id.extend(id_num)
            elif '_filename_0' in cas_table.columns.tolist():
                id_num = cas_table[cas_table['_filename_0'].str.contains(name)]['_id_'].tolist()
                if id_num:
                    image_id.extend(id_num)
    elif isinstance(filename, str):
        if '_path_' in cas_table.columns.tolist():
            image_id = cas_table[cas_table['_path_'].str.contains(filename)]['_id_'].tolist()
        elif '_filename_0' in cas_table.columns.tolist():
            image_id = cas_table[cas_table['_filename_0'].str.contains(filename)]['_id_'].tolist()
    else:
        raise ValueError('filename must be a string or a list of strings')

    if not image_id:
        raise ValueError('filename: {} not found in \'_path_\' or'
                         '\'_filename_0\' columns of table'.format(filename))

    return filter_by_image_id(cas_table, image_id, filtered_name=new_name)


def filter_by_image_id(cas_table, image_id, filtered_name=None):
    '''
    Filter CASTable by '_id_' column

    Parameters
    ----------
    cas_table : CASTable
        Specifies the table to be filtered
        Note: CASTable should have an '_id_' column
    image_id : int or list-of-ints
        Specifies the image id or ids to be kept
    filtered_name : string, optional
        Name of output table

    Returns
    -------
    CASTable
        Filtered table by image_id

    '''
    if filtered_name:
        if isinstance(filtered_name, str):
            new_name = filtered_name
        else:
            raise ValueError('filtered_name must be a str or left as None to generate a random name')
    else:
        new_name = random_name('filtered')

    if '_id_' not in cas_table.columns.tolist():
        raise ValueError('\'_id_\' column not found in CASTable : {}'.format(cas_table.name))
    cas_table = cas_table[cas_table['_id_'].isin(image_id)]
    if cas_table.numrows().numrows == 0:
        raise ValueError('image_id: {} not found in the table'.format(image_id))

    filtered = cas_table.partition(casout=dict(name=new_name, replace=True))['casTable']

    return filtered


def _parse_txt(path):
    input_info = {}
    with open(path, "r") as lines:
        for line in lines:
            info_list = line.replace(' ', '').split('=')
            input_info[info_list[0]] = info_list[1].strip()
    return input_info


def _convert_yolo(size, box):
    ''' Used to normalize bounding box '''
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2])/2.0
    y = (box[1] + box[3])/2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def _convert_coco(size, box, resize):
    w_ratio = float(resize[0]) / size[0]
    h_ratio = float(resize[1]) / size[1]
    x_min = box[0] * w_ratio
    y_min = box[1] * h_ratio
    x_max = box[2] * w_ratio
    y_max = box[3] * h_ratio
    return (x_min, y_min, x_max, y_max)


def _convert_xml_annotation(filename, coord_type, resize):
    in_file = open(filename)
    filename, file_extension = os.path.splitext(filename)
    tree = ET.parse(in_file)
    root = tree.getroot()
    object_ = root.find('object')
    # if a xml is empty, just skip it.
    if object_ is None:
        in_file.close()
        return
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    out_file = open(filename + ".txt", 'w')  # write to test files
    for obj in root.iter('object'):
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        boxes = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        # convert to two formats
        if coord_type == 'yolo':
            boxes = _convert_yolo((width, height), boxes)
        elif coord_type == 'coco':
            boxes = _convert_coco((width, height), boxes, resize)
        out_file.write(str(cls) + "," + ",".join([str(box) for box in boxes]) + '\n')
    in_file.close()
    out_file.close()


def _convert_json_annotation(filename_w_ext, coord_type, resize):
    filename, file_extension = os.path.splitext(filename_w_ext)
    img = Image.open(filename + '.jpg')
    width, height = img.size
    data = json.load(open(filename_w_ext))
    out_file = open('{}.txt'.format(filename), 'w')
    parse_obj = data['detection']['detections']
    for obj in parse_obj:
        cls = obj['type']
        boxes = (obj['ax'], obj['ay'], obj['bx'], obj['by'])
        if coord_type == 'yolo':
            boxes = _convert_yolo((width, height), boxes)
        elif coord_type == 'coco':
            boxes = _convert_coco((width, height), boxes, resize)
        out_file.write(str(cls) + "," + ",".join([str(box) for box in boxes]) + '\n')


def convert_txt_to_xml(path):
    cwd = os.getcwd()
    input_info = _parse_txt(path)
    names = []
    with open(input_info['names'], "r") as lines:
        for line in lines:
            names.append(line)
    os.chdir(input_info['directory'])

    for file in os.listdir():
        filename, file_extension = os.path.splitext(file)
        if file_extension != '.txt' or filename == 'train':
            continue
        img_name = filename + '.jpg'
        img_org = Image.open(filename + '.jpg')
        width_org, height_org = img_org.size
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = path.split('\\')[-1]
        ET.SubElement(annotation, "filename").text = img_name

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(width_org)
        ET.SubElement(size, "height").text = str(height_org)
        ET.SubElement(size, "depth").text = str(3)

        ET.SubElement(annotation, "segmented").text = str(0)

        with open(file, "r") as ins:
            for line in ins:
                object_info = line.split(' ')
                w, h = float(object_info[3]) / 2.0, float(object_info[4]) / 2.0

                xmin = (float(object_info[1]) - w) * float(width_org)
                ymin = (float(object_info[2]) - h) * float(height_org)
                xmax = (float(object_info[1]) + w) * float(width_org)
                ymax = (float(object_info[2]) + h) * float(height_org)

                object = ET.SubElement(annotation, "object")
                ET.SubElement(object, "name").text = names[int(object_info[0])]
                ET.SubElement(object, "pose").text = "Unspecified"
                ET.SubElement(object, "truncated").text = str(0)
                ET.SubElement(object, "occluded").text = str(0)
                ET.SubElement(object, "difficult").text = str(0)
                bndbox = ET.SubElement(object, "bndbox")

                ET.SubElement(bndbox, "xmin").text = str(xmin)
                ET.SubElement(bndbox, "ymin").text = str(ymin)
                ET.SubElement(bndbox, "xmax").text = str(xmax)
                ET.SubElement(bndbox, "ymax").text = str(ymax)

            tree = ET.ElementTree(annotation)
            tree.write(filename + ".xml")
    os.chdir(cwd)


def get_txt_annotation(local_path, coord_type, image_size = (416, 416), label_files = None):
    '''
    Parse object detection annotation files based on Pascal VOC format and save as txt files.

    Parameters
    ----------
    local_path : string
        Local_path points to the directory where xml files are stored.
        The generated txt files will be stored under the directory.
    coord_type : string
        Specifies the type of coordinate to convert into.
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
        'coco' specifies xmin, ymin, xmax, ymax that are borders of a
        bounding boxes.
        The values are relative to parameter image_size.
        Valid Values: yolo, coco
    image_size : tuple or integer, optional
        Specifies the size of images to resize.
        Default: (416, 416)
    label_files : list, optional
        Specifies the list of filename with XML extension under local_path to be parsed.
        If label_files is not specified, all of XML files under local_path will be parsed .
        Default: None

    '''
    cwd = os.getcwd()
    os.chdir(local_path)
    image_size = _pair(image_size)  # ensure image_size is a pair
    # if label_files = None, that means we call it directly and parse annotation files.
    if label_files is None:
        label_files = os.listdir(local_path)
    # find all of label files
    label_files = [x for x in label_files if x.endswith('.xml')]
    if len(label_files) == 0:
        raise DLPyError('Can not find any xml file under data_path')
    for idx, filename in enumerate(label_files):
        _convert_xml_annotation(filename, coord_type, image_size)
    os.chdir(cwd)


def create_object_detection_table(conn, data_path, coord_type, output,
                                  local_path=None, image_size=(416, 416)):
    '''
    Create an object detection table

    Parameters
    ----------
    conn : session
        CAS connection object
    data_path : string
        Specifies a location where annotation files and image files are stored.
        Annotation files should be XML file based on Pascal VOC format
        Notice that the path should be accessible by CAS server.
    coord_type : string
        Specifies the type of coordinate to convert into.
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
        'coco' specifies xmin, ymin, xmax, ymax that are borders of a
        bounding boxes.
        The values are relative to parameter image_size.
        Valid Values: yolo, coco
    output : string
        Specifies the name of the object detection table.
    local_path : string, optional
        Local_path and data_path point to the same location.
        The parameter local_path will be optional (default=None) if the
        Python client has the same OS as CAS server or annotation files
        in TXT format are placed in data_path.
        Otherwise, the path that depends on the Python client OS needs to be specified.
        For example:
        Windows client with linux CAS server:
        data_path=/path/to/data/path
        local_path=\\path\to\data\path
        Linux clients with Windows CAS Server:
        data_path=\\path\to\data\path
        local_path=/path/to/data/path
    image_size : tuple or integer, optional
        Specifies the size of images to resize.
        If a tuple is passed, the first integer is width and the second value is height.
        Default: (416, 416)

    Returns
    -------
    A list of variables that are the labels of the object detection table

    '''
    if coord_type.lower() not in ['yolo', 'coco']:
        raise ValueError('coord_type, {}, is not supported'.format(coord_type))
    with sw.option_context(print_messages=False):
        server_type = get_cas_host_type(conn).lower()
    local_os_type = platform.system()
    unix_type = server_type.startswith("lin") or server_type.startswith("osx")
    # check if local and server are same type of OS
    # in different os
    need_to_parse = True
    if (unix_type and local_os_type.startswith('Win')) or not (unix_type or local_os_type.startswith('Win')):
        if local_path is None:
            print('The txt files in data_path are used as annotation files.')
            need_to_parse = False
    else:
        local_path = data_path

    conn.retrieve('loadactionset', _messagelevel='error', actionset='image')
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'deepLearn')
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'transpose')

    # label variables, _ : category;
    yolo_var_name = ['_', '_x', '_y', '_width', '_height']
    coco_var_name = ['_', '_xmin', '_ymin', '_xmax', '_ymax']
    if coord_type.lower() == 'yolo':
        var_name = yolo_var_name
    elif coord_type.lower() == 'coco':
        var_name = coco_var_name
    image_size = _pair(image_size)  # ensure image_size is a pair
    det_img_table = random_name('DET_IMG')

    caslib, path_after_caslib, tmp_caslib = caslibify(conn, data_path, task='load')
    if caslib is None and path_after_caslib is None:
        print('Cannot create a caslib for the provided path. Please make sure that the path is accessible from'
              'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')

    with sw.option_context(print_messages=False):
        res = conn.image.loadImages(path=path_after_caslib,
                                    recurse=False,
                                    labelLevels=-1,
                                    caslib=caslib,
                                    casout={'name': det_img_table, 'replace':True})
        if res.severity > 0:
            for msg in res.messages:
                if not msg.startswith('WARNING'):
                    print(msg)

        res = conn.image.processImages(table={'name': det_img_table},
                                       imagefunctions=[
                                           {'options': {'functiontype': 'RESIZE',
                                                        'height': image_size[1],
                                                        'width': image_size[0]}}
                                       ],
                                       casout={'name': det_img_table, 'replace': True})

        if res.severity > 0:
            for msg in res.messages:
                print(msg)
        else:
            print("NOTE: Images are processed.")

    if (caslib is not None) and tmp_caslib:
        conn.retrieve('dropcaslib', _messagelevel='error', caslib=caslib)

    with sw.option_context(print_messages = False):
        caslib = find_caslib(conn, data_path)
        if caslib is None:
            caslib = random_name('Caslib', 6)
            rt = conn.retrieve('addcaslib', _messagelevel = 'error', name = caslib, path = data_path,
                               activeonadd = False, subdirectories = True, datasource = {'srctype': 'path'})
            if rt.severity > 1:
                raise DLPyError('something went wrong while adding the caslib for the specified path.')

    # find all of annotation files under the directory
    label_files = conn.fileinfo(caslib = caslib, allfiles = True).FileInfo['Name'].values
    # if client and server are on different type of operation system, we assume user parse xml files and put
    # txt files in data_path folder. So skip get_txt_annotation()
    # parse xml or json files and create txt files
    if need_to_parse:
        get_txt_annotation(local_path, coord_type, image_size, label_files)

    label_tbl_name = random_name('obj_det')
    # load all of txt files into cas server
    label_files = conn.fileinfo(caslib = caslib, allfiles = True).FileInfo['Name'].values
    label_files = [x for x in label_files if x.endswith('.txt')]
    if len(label_files) == 0:
        raise DLPyError('Can not find any txt file under data_path.')
    idjoin_format_length = len(max(label_files, key=len)) - len('.txt')
    with sw.option_context(print_messages = False):
        for idx, filename in enumerate(label_files):
            tbl_name = '{}_{}'.format(label_tbl_name, idx)
            conn.retrieve('loadtable', caslib = caslib, path = filename,
                          casout = dict(name = tbl_name, replace = True),
                          importOptions = dict(fileType = 'csv', getNames = False,
                                               varChars = True, delimiter = ','))
            conn.retrieve('partition',
                          table = dict(name = tbl_name,
                                       compvars = ['idjoin'],
                                       comppgm = 'length idjoin $ {};idjoin="{}";'.format(idjoin_format_length,
                                                                                          filename[:-len('.txt')])),
                          casout = dict(name = tbl_name, replace = True))

    input_tbl_name = ['{}_{}'.format(label_tbl_name, i) for i in range(idx + 1)]
    string_input_tbl_name = ' '.join(input_tbl_name)
    # concatenate all of annotation table together
    fmt_code = '''
                data {0}; 
                set {1}; 
                run;
                '''.format(output, string_input_tbl_name)
    conn.runcode(code = fmt_code, _messagelevel = 'error')
    cls_col_format_length = conn.columninfo(output).ColumnInfo.loc[0][3]
    cls_col_format_length = cls_col_format_length if cls_col_format_length >= len('NoObject') else len('NoObject')

    conn.altertable(name = output, columns = [dict(name = 'Var1', rename = var_name[0]),
                                              dict(name = 'Var2', rename = var_name[1]),
                                              dict(name = 'Var3', rename = var_name[2]),
                                              dict(name = 'Var4', rename = var_name[3]),
                                              dict(name = 'Var5', rename = var_name[4])])
    # add sequence id that is used to build column name in transpose process
    sas_code = '''
               data {0};
                  set {0} ;
                  by idjoin;
                  seq_id+1;
                  if first.idjoin then seq_id=0;
                  output;
               run;
               '''.format(output)
    conn.runcode(code = sas_code, _messagelevel = 'error')
    # convert long table to wide table
    with sw.option_context(print_messages = False):
        for var in var_name:
            conn.transpose(prefix = '_Object', suffix = var, id = 'seq_id', transpose = [var],
                           table = dict(name = output, groupby = 'idjoin'),
                           casout = dict(name = 'output{}'.format(var), replace = 1))
            conn.altertable(name = 'output{}'.format(var), columns=[{'name': '_NAME_', 'drop': True}])
    # dljoin the five label columns
    res = conn.deeplearn.dljoin(table = 'output{}'.format(var_name[0]), id = 'idjoin',
                                annotatedtable = 'output{}'.format(var_name[1]),
                                casout = dict(name = output, replace = True), _messagelevel = 'error')
    if res.severity > 0:
        raise DLPyError('ERROR: Fail to create the object detection table.')

    for var in var_name[2:]:
        res = conn.deepLearn.dljoin(table = output, id = 'idjoin', annotatedtable = 'output{}'.format(var),
                                    casout = dict(name = output, replace = True))
        if res.severity > 0:
            raise DLPyError('ERROR: Fail to create the object detection table.')
    # get number of objects in each image
    code = '''
            data {0};
            set {0};
            array _all _numeric_;
            _nObjects_ = (dim(_all)-cmiss(of _all[*]))/4;
            run;
            '''.format(output)
    conn.runcode(code = code, _messagelevel = 'error')
    max_instance = int((max(conn.columninfo(output).ColumnInfo['ID'])-1)/5)
    var_order = ['idjoin', '_nObjects_']
    for i in range(max_instance):
        for var in var_name:
            var_order.append('_Object'+str(i)+var)
    # change order of columns and unify the formattedlength of class columns
    format_ = '${}.'.format(cls_col_format_length)
    res = conn.altertable(name = output, columns = [{'name': '_Object{}_'.format(i), 'format': format_}
                                                    for i in range(max_instance)])
    if res.severity > 0:
        raise DLPyError('ERROR: Fail to create the object detection table.')

    # parse and create dljoin id column
    label_col_info = conn.columninfo(output).ColumnInfo
    filename_col_length = label_col_info.loc[label_col_info['Column'] == 'idjoin', ['FormattedLength']].values[0][0]

    image_sas_code = "length idjoin $ {0}; fn=scan(_path_,{1},'/'); idjoin = inputc(substr(fn, 1, length(fn)-4),'{0}.');".format(filename_col_length,
                                                len(data_path.split('\\')) - 2)
    img_tbl = conn.CASTable(det_img_table,
                            computedvars = ['idjoin'],
                            computedvarsprogram = image_sas_code,
                            vars = [{'name': '_image_'}])

    # join the image table and label table together
    res = conn.deepLearn.dljoin(table = img_tbl, annotation = output, id = 'idjoin',
                                casout = {'name': output, 'replace': True, 'replication': 0})
    if res.severity > 0:
        raise DLPyError('ERROR: Fail to create the object detection table.')

    with sw.option_context(print_messages=False):
        for name in input_tbl_name:
            conn.table.droptable(name)
        for var in var_name:
            conn.table.droptable('output{}'.format(var))
        conn.table.droptable(det_img_table)

    conn.retrieve('dropcaslib', _messagelevel='error', caslib=caslib)

    print("NOTE: Object detection table is successfully created.")
    return var_order[2:]


def display_object_detections(conn, table, coord_type, max_objects=10,
                              num_plot=10, n_col=2, fig_size=None):
    '''
    Plot images with drawn bounding boxes.

    conn : CAS
        CAS connection object
    table : string or CASTable
        Specifies the object detection castable to be plotted.
    coord_type : string
        Specifies coordinate type of input table
    max_objects : int, optional
        Specifies the maximum number of bounding boxes to be plotted on an image.
        Default: 10
    num_plot : int, optional
        Specifies the name of the castable.
    n_col : int, optional
        Specifies the number of column to plot.
        Default: 2
    fig_size : int, optional
        Specifies the size of figure.

    '''
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'image')

    input_tbl_opts = input_table_check(table)
    input_table = conn.CASTable(**input_tbl_opts)
    img_num = input_table.shape[0]
    num_plot = num_plot if num_plot < img_num else img_num
    input_table = input_table.sample(num_plot)
    det_label_image_table = random_name('detLabelImageTable')

    num_max_obj = input_table['_nObjects_'].max()
    max_objects = max_objects if num_max_obj > max_objects else num_max_obj

    with sw.option_context(print_messages=False):
        res = conn.image.extractdetectedobjects(casout = {'name': det_label_image_table, 'replace': True},
                                                coordtype=coord_type,
                                                maxobjects=max_objects,
                                                table=input_table)
        if res.severity > 0:
            for msg in res.messages:
                print(msg)

    outtable = conn.CASTable( det_label_image_table)
    num_detection = len(outtable)
    # print('{} out of {} images have bounding boxes to display'.format(num_detection, img_num))
    if num_detection == 0:
        print('Since there is no image that contains a bounding box, cannot display any image.')
        return
    num_plot = num_plot if num_plot < num_detection else num_detection
    # if random_plot:
    #     conn.shuffle(det_label_image_table, casout = {'name': det_label_image_table, 'replace': True})

    with sw.option_context(print_messages=False):
        prediction_plot = conn.image.fetchImages(imageTable = {'name': det_label_image_table},
                                                 to = num_plot,
                                                 fetchImagesVars = ['_image_', '_path_'])
        if res.severity > 0:
            for msg in res.messages:
                print(msg)

    if num_plot > n_col:
        n_row = num_plot // n_col + 1
    else:
        n_row = 1
        n_col = num_plot

    n_col_m = n_col
    if n_col_m < 1:
        n_col_m += 1

    n_row_m = n_row
    if n_row < 1:
        n_row_m += 1

    if fig_size is None:
        fig_size = (16, 16 // n_col_m * n_row_m)

    fig = plt.figure(figsize = fig_size)

    k = 1

    for i in range(num_plot):
        image = prediction_plot['Images']['Image'][i]
        ax = fig.add_subplot(n_row, n_col, k)
        plt.imshow(image)
        if '_path_' in prediction_plot['Images'].columns:
            plt.title(str(os.path.basename(prediction_plot['Images']['_path_'].loc[i])))
        k = k + 1
        plt.xticks([]), plt.yticks([])
    plt.show()

    with sw.option_context(print_messages=False):
        conn.table.droptable(det_label_image_table)


def plot_anchors(base_anchor_size, anchor_scale, anchor_ratio, image_size, fig_size=(10, 10)):
    '''
    Plot proposed anchor boxes in Region Proposal Layer

    Parameters
    ----------
    base_anchor_size : int, optional
        Specifies the basic anchor size in width and height (in pixels) in the original input image dimension
        Default: 16
    anchor_ratio : iter-of-float
        Specifies the anchor height and width ratios (h/w) used.
    anchor_scale : iter-of-float
        Specifies the anchor scales used based on base_anchor_size.
    image_size : iter-of-int
        Specifies the shape of input images in two dimensions(h, w), such as (496, 1000).
    fig_size : int, optional
        Specifies the size of figure.

    '''
    # color map to draw anchor boxes
    color_map = ['b', 'g', 'r', 'c', 'm', 'y']
    img_height = image_size[0]
    img_width = image_size[1]
    anchors = []
    max_anchor_height = image_size[0]
    max_anchor_width = image_size[1]
    # generate all of anchors based on base_anchor_size, anchor scale and anchor ratio
    for ratio in anchor_ratio:
        for scale in anchor_scale:
            len_size = base_anchor_size * scale
            area = len_size * len_size
            height = math.sqrt(area * ratio)
            width = height / ratio
            anchors.append((height, width))
    # get background height/width that is the largest value in the shape of the image and the largest anchor box.
    for an in anchors:
        max_anchor_height = max(max_anchor_height, an[0])
        max_anchor_width = max(max_anchor_width, an[1])
    fig, ax = plt.subplots(1, figsize = fig_size)
    plt.xticks([]), plt.yticks([])
    # draw the background
    background = np.tile((255, 255, 255), (int(max_anchor_height), int(max_anchor_width), 1))
    # draw the image region
    image_region = (int((max_anchor_height - img_height) / 2), int((max_anchor_height + img_height) / 2),
                    int((max_anchor_width - img_width) / 2), int((max_anchor_width + img_width) / 2))
    background[image_region[0]: image_region[1], image_region[2]: image_region[3], :] = np.array((244, 203, 66))
    ax.imshow(background)
    # draw the anchor boxes
    for i, anchor in enumerate(anchors):
        centric_x = (max_anchor_width - anchor[1]) / 2  # x
        centric_y = (max_anchor_height - anchor[0]) / 2  # y
        color = color_map[i % len(anchor_scale)]
        rect = patches.Rectangle((centric_x, centric_y), anchor[1], anchor[0], linewidth = 2,
                                 edgecolor = color, facecolor = 'none')
        ax.add_patch(rect)


def get_mapping_dict():
    project_path = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join('datasources', 'mapping.json')
    full_filename = os.path.join(project_path, filename)
    with open(full_filename) as f:
        j = json.load(f)
    return j


def char_to_double(conn, tbl_colinfo, input_tbl_name, 
                   output_tbl_name, varlist, num_fmt='8.'):
    varlist_lower = [var.lower() for var in varlist]
    
    fmt_list = tbl_colinfo.loc[(
            (tbl_colinfo.Column.str.lower().isin(varlist_lower)) &
             (tbl_colinfo.Type != 'double')
             ),'Column'].tolist()

    int_list = tbl_colinfo.loc[(
            (tbl_colinfo.Column.str.lower().isin(varlist_lower)) &
             (tbl_colinfo.Type.str.startswith('int'))
             ),'Column'].tolist()

    char_list = [var for var in fmt_list if var not in int_list]

    if len(char_list) > 0:
        fmt_code = '''
        data {0};
        set {1}(rename=(
        '''.format(output_tbl_name, input_tbl_name)
                  
        for var in char_list:
            fmt_code += '{0}=c_{0} '.format(var) #The space is important
            
        fmt_code += '));'
        
        for var in char_list:
            fmt_code += '''
            {0} = input(c_{0},{1});
            drop c_{0};         
            '''.format(var, num_fmt)
            
        fmt_code += 'run;'          
    else:
        fmt_code = '''
        data {0};
        set {1};
        run;
        '''.format(output_tbl_name, input_tbl_name)            
    
    conn.retrieve('dataStep.runCode', _messagelevel='error', code=fmt_code)
    

def int_to_double(conn, tbl_colinfo, input_tbl_name, 
                  output_tbl_name, varlist, num_fmt='8.'):
    varlist_lower = [var.lower() for var in varlist]

    int_list = tbl_colinfo.loc[(
            (tbl_colinfo.Column.str.lower().isin(varlist_lower)) &
             (tbl_colinfo.Type.str.startswith('int'))
             ),'Column'].tolist()

    if len(int_list) > 0:
        fmt_code = '''
        data {0};
        set {1};
        '''.format(output_tbl_name, input_tbl_name)

        for var in int_list:
            fmt_code += '''
            format {0} {1};       
            '''.format(var, num_fmt)
            
        fmt_code += 'run;'    
    else:
        fmt_code = '''
        data {0};
        set {1};
        run;
        '''.format(output_tbl_name, input_tbl_name)            

    conn.retrieve('dataStep.runCode', _messagelevel='error', code=fmt_code)


def display_segmentation_images(conn, table, n_images=4, image_column='_image_',
                                segmentation_labels_table=None, label_column='labels',
                                fig_size=(50, 20)):
    '''

    This function is designed to display images of a castable. It also displays the segmentation labels
    if it is set. Note that the ground truth (i.e., label) information in the segmentation task is also images.
    On top of displaying images and labels, this function is also flexible enough to display the predictions.
    It is always in the following order: images from the table table, images from the segmentation labels table
    (if any), and images from the segmentation prediction table (if any).

    conn : CAS
        CAS connection object
    table : string or CASTable
        Specifies the input table that has an image column.
    n_images : int
        Specifies the number of images to be displayed.
    image_column : string
        Specifies the column name that holds the image data in the input table.
    segmentation_labels_table : string or CASTable
        Specifies the table that has the segmentation labels (or label images).
    label_column : string
        Specifies the name of the column that holds the segmentation labels in the segmentation labels table.
    fig_size : a list of two ints
        Specifies the figure size.

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='image')
    if not isinstance(table, CASTable):
        tbl = conn.CASTable(table)
    else:
        tbl = table

    images = conn.retrieve('image.fetchImages', _messagelevel='error',
                           table=tbl,
                           to=n_images,
                           image=image_column)

    if len(images.Images) < 1:
        raise DLPyError('input table does not have any images')

    labels = None
    if segmentation_labels_table is not None:
        if not isinstance(segmentation_labels_table, CASTable):
            seg_gt_tbl = conn.CASTable(segmentation_labels_table)
        else:
            seg_gt_tbl = segmentation_labels_table

        labels = conn.retrieve('image.fetchImages', _messagelevel='none',
                               table=seg_gt_tbl,
                               to=n_images,
                               image=label_column)
        if len(labels) == 0 or len(labels.Images) == 0:
            print('WARNING: Something went wrong while extracting label images')

    k = 1
    n_row = 2
    fig = plt.figure(figsize=fig_size)

    for i in range(n_images):
        ax = fig.add_subplot(n_row, n_images, k)
        plt.imshow(images['Images']['Image'][i])
        k += 1
    if len(labels) > 0 and len(labels.Images) > 0:
        for i in range(n_images):
            ax = fig.add_subplot(n_row, n_images, k)
            plt.imshow(np.array(labels['Images']['Image'][i])[:, :, 0], vmax=2)
            k += 1


def display_segmentation_results(conn, table, n_images=4, image_column='_image_',
                                 segmentation_labels_table=None, label_column='labels',
                                 segmentation_prediction_table=None, prediction_column=None,
                                 filename_column=None,
                                 fig_size=(15, 40)):
    '''

    This function is designed to display images of a castable. It also displays the segmentation labels
    if it is set. Note that the ground truth (i.e., label) information in the segmentation task is also images.
    On top of displaying images and labels, this function is also flexible enough to display the predictions.

    conn : CAS
        CAS connection object
    table : string or CASTable
        Specifies the input table that has an image column.
    n_images : int
        Specifies the number of images to be displayed.
    image_column : string
        Specifies the column name that holds the image data in the input table.
    segmentation_labels_table : string or CASTable
        Specifies the table that has the segmentation labels (or label images).
    label_column : string
        Specifies the name of the column that holds the segmentation labels in the segmentation labels table.
    segmentation_prediction_table : string or CASTable
        Specifies the table that has the segmentation predictions.
    prediction_column : string
        Specifies the name of the column that holds the segmentation predictions in the segmentation prediction table.
    filename_column : string
        Specifies the name of the column that holds the filenames of the images. If this column set, the column names
        will be displayed on top of each image.
    fig_size : a list of two ints
        Specifies the figure size.

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='image')
    if not isinstance(table, CASTable):
        tbl = conn.CASTable(table)
    else:
        tbl = table

    images = conn.retrieve('image.fetchImages', _messagelevel='error',
                           table=tbl,
                           to=n_images,
                           image=image_column)

    if len(images.Images) < 1:
        raise DLPyError('input table does not have any images')

    n_col = 1
    labels = None
    if segmentation_labels_table is not None:
        if not isinstance(segmentation_labels_table, CASTable):
            seg_gt_tbl = conn.CASTable(segmentation_labels_table)
        else:
            seg_gt_tbl = segmentation_labels_table

        labels = conn.retrieve('image.fetchImages', _messagelevel='none',
                               table=seg_gt_tbl,
                               to=n_images,
                               image=label_column)
        if len(labels) == 0 or len(labels.Images) == 0:
            print('WARNING: Something went wrong while extracting label images')
        else:
            n_col += 1

    predictions = None
    if segmentation_prediction_table is not None:

        if prediction_column is None:
            raise DLPyError('Please set the prediction_column parameter')

        if not isinstance(segmentation_prediction_table, CASTable):
            seg_prediction_tbl = conn.CASTable(segmentation_prediction_table)
        else:
            seg_prediction_tbl = segmentation_prediction_table

        if filename_column is not None:
            predictions = conn.retrieve('image.fetchImages', _messagelevel='none',
                                        table=seg_prediction_tbl,
                                        to=n_images,
                                        fetchImagesVars=filename_column,
                                        image=prediction_column)
        else:
            predictions = conn.retrieve('image.fetchImages', _messagelevel='none',
                                        table=seg_prediction_tbl,
                                        to=n_images,
                                        image=prediction_column)

        if len(predictions) == 0 or len(predictions.Images) == 0:
            print('WARNING: Something went wrong while extracting output (predicted) images')
        else:
            n_col += 1

    k = 1
    fig = plt.figure(figsize=fig_size)

    for i in range(n_images):
        ax = fig.add_subplot(n_images, n_col, k)
        plt.imshow(images['Images']['Image'][i])
        k += 1
        if len(predictions) > 0 and len(predictions.Images) > 0:
            plt.title(predictions.Images[filename_column][i] +' raw image')
            plt.xticks([]), plt.yticks([])

        if len(labels) > 0 and len(labels.Images) > 0:
            ax = fig.add_subplot(n_images, n_col, k)
            plt.imshow(np.array(labels['Images']['Image'][i])[:, :, 0], vmax=2)
            k += 1
            if len(predictions) > 0 and len(predictions.Images) > 0:
                plt.title(predictions.Images[filename_column][i] +' ground truth')
                plt.xticks([]), plt.yticks([])

        if len(predictions) > 0 and len(predictions.Images) > 0:
            ax = fig.add_subplot(n_images, n_col, k)
            plt.imshow(np.array(predictions['Images']['Image'][i]))
            k += 1
            if len(predictions) > 0 and len(predictions.Images) > 0:
                plt.title(predictions.Images[filename_column][i] +' prediction')
                plt.xticks([]), plt.yticks([])


def create_object_detection_table_no_xml(conn, data_path, coord_type, output, annotation_path=None,
                                         image_size=(416, 416)):
    '''
    This is an alternative function to create object detection table. This function is especially good if you are
    using Ethem's annotation tool (this one creates txt files directly).

    Parameters
    ----------
    conn : session
        CAS connection object
    data_path : string
        Specifies a location where annotation files and image files are stored.
        Annotation files should be XML file based on Pascal VOC format
        Notice that the path should be accessible by CAS server.
    coord_type : string
        Specifies the type of coordinate to convert into.
        'yolo' specifies x, y, width and height, x, y is the center
        location of the object in the grid cell. x, y, are between 0
        and 1 and are relative to that grid cell. x, y = 0,0 corresponds
        to the top left pixel of the grid cell.
        'coco' specifies xmin, ymin, xmax, ymax that are borders of a
        bounding boxes.
        The values are relative to parameter image_size.
        Valid Values: yolo, coco
    output : string
        Specifies the name of the object detection table.
    annotation_path: string
        Specifies the location of the annotations. This folder needs to be accessed by either the CAS server
        or DLPy.
    image_size : tuple or integer, optional
        Specifies the size of images to resize.
        If a tuple is passed, the first integer is width and the second value is height.
        Default: (416, 416)

    Returns
    -------
    A list of variables that are the labels of the object detection table

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='image')
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'deepLearn')
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'transpose')

    # now it is time to check if we can read the annotation files from the server
    caslib_annotation = None
    annotation_data_is_in_the_client = 0
    label_files = []
    with sw.option_context(print_messages=False):
        caslib_annotation, path_after_ann_caslib, tmp_caslib = caslibify(conn, annotation_path, task='save')
        if caslib_annotation is None:
            caslib_annotation = random_name('Caslib', 6)
            rt = conn.retrieve('addcaslib', _messagelevel = 'error', name = caslib_annotation, path = annotation_path,
                               activeonadd=False, subdirectories = True, datasource = {'srctype': 'path'})
            if rt.severity > 1:
                print('NOTE: Something went wrong while adding the caslib for the specified path. this might '
                      'be due to the following reasons: 1) server cannot access the annotation_path, '
                      '2) you do not have permission to use this path, '
                      '3) there is already a caslib on one of the subpaths')
                print('NOTE: Now we will check if DLPy can access the annotation_path')

                try:
                    import os
                    if not os.path.isdir(annotation_path) and not os.path.isfile(annotation_path):
                        raise DLPyError('DLPy cannot access the data as well. Please make sure the annotation data is'
                                        'accessible from DLPy or the server')

                    annotation_data_is_in_the_client = 1
                    for root, dirs, files in os.walk(annotation_path):
                        d = os.path.basename(root)
                        for file in files:
                            if file.endswith('.txt'):
                                label_files.append(os.path.join(d, file))
                except:
                    raise DLPyError('DLPy cannot access the data as well. Please make sure the annotation data is'
                                    'accessible from DLPy or the server')
            else:
                label_files = conn.fileinfo(caslib =caslib_annotation, allfiles=True).FileInfo['Name'].values
                label_files = [x for x in label_files if x.endswith('.txt')]
        else:
            label_files = conn.fileinfo(caslib=caslib_annotation, allfiles=True).FileInfo['Name'].values
            label_files = [x for x in label_files if x.endswith('.txt')]

    if len(label_files) == 0:
        raise DLPyError('There is no annotation file in the annotation_path.')

    if (caslib_annotation is not None) and tmp_caslib:
        conn.retrieve('dropcaslib', _messagelevel='error', caslib=caslib_annotation)

    caslib, path_after_caslib, tmp_caslib = caslibify(conn, data_path, task='load')
    if caslib is None and path_after_caslib is None:
        print('Cannot create a caslib for the provided (i.e., '+data_path+') path. Please make sure that the '
                                                                          'path is accessible from'
                                                                          'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')
    det_img_table = random_name('DET_IMG')
    image_size = _pair(image_size)  # ensure image_size is a pair
    with sw.option_context(print_messages=False):
        res = conn.image.loadImages(path=path_after_caslib,
                                    recurse=True,
                                    labelLevels=-1,
                                    caslib=caslib,
                                    casout={'name': det_img_table, 'replace': True})
        if res.severity > 0:
            for msg in res.messages:
                if not msg.startswith('WARNING'):
                    print(msg)
        else:
            print('NOTE: Images are loaded.')

        res = conn.image.processImages(table={'name': det_img_table},
                                       imagefunctions=[
                                           {'options': {'functiontype': 'RESIZE',
                                                        'height': image_size[1],
                                                        'width': image_size[0]}}
                                       ],
                                       casout={'name': det_img_table, 'replace': True})

        if res.severity > 0:
            for msg in res.messages:
                print(msg)
        else:
            print("NOTE: Images are processed.")

    if (caslib is not None) and tmp_caslib:
        conn.retrieve('dropcaslib', _messagelevel='error', caslib=caslib)
        caslib=None

    if coord_type.lower() not in ['yolo', 'coco']:
        raise ValueError('coord_type, {}, is not supported'.format(coord_type))

    # label variables, _ : category;
    yolo_var_name = ['_', '_x', '_y', '_width', '_height']
    coco_var_name = ['_', '_xmin', '_ymin', '_xmax', '_ymax']
    if coord_type.lower() == 'yolo':
        var_name = yolo_var_name
    elif coord_type.lower() == 'coco':
        var_name = coco_var_name

    if annotation_data_is_in_the_client == 0:
        caslib_annotation, path_after_ann_caslib, tmp_caslib = caslibify(conn, annotation_path, task='save')
    else:
        tmp_caslib = False

    label_tbl_name = random_name('obj_det')
    idjoin_format_length = len(max(label_files, key=len)) - len('.txt')
    with sw.option_context(print_messages=False):
        for idx, filename in enumerate(label_files):
            tbl_name = '{}_{}'.format(label_tbl_name, idx)
            if annotation_data_is_in_the_client:
                rt = conn.retrieve('upload', path=filename, casout=dict(name=tbl_name, replace=True),
                                   importOptions=dict(fileType='csv', getNames=False, varChars=True, delimiter=','))
                if rt.severity > 1:
                    raise DLPyError('DLPy cannot upload the ' + filename + ' to the server.')
            else:
                conn.retrieve('loadtable', caslib=caslib_annotation, path=filename,
                              casout = dict(name=tbl_name, replace=True),
                              importOptions=dict(fileType='csv', getNames=False, varChars=True, delimiter=','))

            conn.retrieve('partition',
                          table=dict(name=tbl_name, compvars=['idjoin'],
                                     comppgm='length idjoin $ {};idjoin="{}";'.format(idjoin_format_length,
                                                                                      filename[:-len('.txt')])),
                          casout=dict(name=tbl_name, replace=True))

    input_tbl_name = ['{}_{}'.format(label_tbl_name, i) for i in range(idx + 1)]
    string_input_tbl_name = ' '.join(input_tbl_name)
    # concatenate all of annotation table together
    fmt_code = '''
                data {0}; 
                set {1}; 
                run;
                '''.format(output, string_input_tbl_name)
    conn.runcode(code=fmt_code, _messagelevel='error')
    cls_col_format_length = conn.columninfo(output).ColumnInfo.loc[0][3]
    cls_col_format_length = cls_col_format_length if cls_col_format_length >= len('NoObject') else len('NoObject')

    print('labels are being processed')

    conn.altertable(name=output, columns=[dict(name='Var1', rename=var_name[0]),
                                          dict(name='Var2', rename=var_name[1]),
                                          dict(name='Var3', rename=var_name[2]),
                                          dict(name='Var4', rename=var_name[3]),
                                          dict(name='Var5', rename=var_name[4])])
    # add sequence id that is used to build column name in transpose process
    sas_code = '''
               data {0};
                  set {0} ;
                  by idjoin;
                  seq_id+1;
                  if first.idjoin then seq_id=0;
                  output;
               run;
               '''.format(output)
    conn.runcode(code = sas_code, _messagelevel = 'error')
    # convert long table to wide table
    with sw.option_context(print_messages = False):
        for var in var_name:
            conn.transpose(prefix='_Object', suffix=var, id='seq_id', transpose=[var],
                           table=dict(name=output, groupby='idjoin'),
                           casout=dict(name='output{}'.format(var), replace=1))
            conn.altertable(name='output{}'.format(var), columns=[{'name': '_NAME_', 'drop': True}])
    # dljoin the five columns
    res = conn.deeplearn.dljoin(table='output{}'.format(var_name[0]), id='idjoin',
                          annotatedtable='output{}'.format(var_name[1]),
                          casout=dict(name=output, replace=True), _messagelevel='error')
    if res.severity > 0:
        raise DLPyError('ERROR: Fail to create the object detection table.')

    for var in var_name[2:]:
        res = conn.deepLearn.dljoin(table=output, id='idjoin', annotatedtable='output{}'.format(var),
                              casout=dict(name=output, replace=True))
        if res.severity > 0:
            raise DLPyError('ERROR: Fail to create the object detection table.')
    # get number of objects in each image
    code = '''
            data {0};
            set {0};
            array _all _numeric_;
            _nObjects_ = (dim(_all)-cmiss(of _all[*]))/4;
            run;
            '''.format(output)
    conn.runcode(code=code, _messagelevel='error')
    max_instance = int((max(conn.columninfo(output).ColumnInfo['ID'])-1)/5)
    var_order = ['idjoin', '_nObjects_']
    for i in range(max_instance):
        for var in var_name:
            var_order.append('_Object'+str(i)+var)
    # change order of columns and unify the formattedlength of class columns
    format_ = '${}.'.format(cls_col_format_length)
    # conn.altertable(name = output, columnorder = var_order, columns =[{'name': '_Object{}_'.format(i),
    #                                                                    'format': format_} for i in range(max_instance)])
    conn.altertable(name=output, columns=[{'name': '_Object{}_'.format(i), 'format': format_}
                                          for i in range(max_instance)])
    # parse and create dljoin id column
    label_col_info = conn.columninfo(output).ColumnInfo
    filename_col_length = label_col_info.loc[label_col_info['Column'] == 'idjoin', ['FormattedLength']].values[0][0]

    image_sas_code = "length idjoin $ {0}; fn=scan(_path_,{1},'/'); idjoin = inputc(substr(fn, 1, length(fn)-4),'{0}.');".format(filename_col_length,
                                                                                                                                 len(data_path.split('\\')) - 2)
    img_tbl = conn.CASTable(det_img_table, computedvars=['idjoin'], computedvarsprogram=image_sas_code, vars=[{'name': '_image_'}])

    # join the image table and label table together
    res = conn.deepLearn.dljoin(table=img_tbl, annotation=output, id='idjoin',
                                casout={'name': output, 'replace': True, 'replication': 0})
    if res.severity > 0:
        raise DLPyError('ERROR: Fail to create the object detection table.')

    with sw.option_context(print_messages=False):
        for name in input_tbl_name:
            conn.table.droptable(name)
        for var in var_name:
            conn.table.droptable('output{}'.format(var))
        conn.table.droptable(det_img_table)

    if (caslib_annotation is not None) and tmp_caslib:
        conn.retrieve('dropcaslib', _messagelevel='error', caslib=caslib_annotation)

    print("NOTE: Object detection table is successfully created.")
    return var_order[2:]


def _ntuple(n):
    '''
    create a function used to generate a tuple with length of n

    n : int
        specifies the length of the tuple.

    '''
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)
_triple = _ntuple(3)


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def parameter_2d(param1, param2, param3, default_value):
    '''
    a help function to generate layer properties such as strides, padding, output_padding.

    For example:
        parameter_2d(param1=None, param2=None, param3=None, default_value=(4, 4)) would return (4, 4)
        parameter_2d(param1=2, param2=3, param3=2, default_value=(4, 4)) would return (3, 2)
        parameter_2d(param1=2, param2=None, param3=None, default_value=(4, 4)) would return (2, 2)
        parameter_2d(param1=None, param2=None, param3=2, default_value=(4, 4)) would return (4, 2)

    param1 : int
        specifies the layer option, such as stride, padding, output_padding
    param2 : int
        specifies the layer option related to the first dimension, such as stride_vertical, padding_height
    param3 : int
        specifies the layer option related to the second dimension, such as stride_horizontal, padding_width
    default_value : tuple
        specifies default value

    '''
    if param1 is not None:
        return _pair(param1)
    elif not any([param2, param3]):
        return default_value
    else:
        if param2 is None:
            return (default_value[0], param3)
        if param3 is None:
            return (param2, default_value[1])
        else:
            return (param2, param3)


class DLPyDict(collections.MutableMapping):
    """ Dictionary that applies an arbitrary key-altering function before accessing the keys """

    def __init__(self, *args, **kwargs):
        for k in kwargs:
            self.__setitem__(k, kwargs[k])

    def __getitem__(self, key):
        return self.__dict__[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if value is not None:
            self.__dict__[self.__keytransform__(key)] = value
        else:
            if key in self.__dict__:
                self.__delitem__[key]

    def __delitem__(self, key):
        del self.__dict__[self.__keytransform__(key)]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __keytransform__(self, key):
        return key.lower().replace("_", "")

    def __str__(self):
        return str(self.__dict__)
