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

''' Special functionality for image embedding tables containing image data '''
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from swat import get_option, set_option
from swat.cas.table import CASTable

from dlpy import ImageTable
from .utils import random_name, image_blocksize, caslibify_context, get_server_path_sep, get_cas_host_type, DLPyError, \
    file_exist_on_server
from warnings import warn


class ImageEmbeddingTable(ImageTable):
    '''

    Specialized CASTable for Image Embedding Data. For Siamese model, this table contains two image columns.
    Each image pair can be either (P, P) or (P, N). P means positive and N means negative.
    For triplet model, it contains three image columns, i.e., (A, P, N)
    while for quartet model, it contains four image columns, i.e., (A, P, N, N1). A means anchor.

    Parameters
    ----------
    name : string
        The name of the CAS table
    **table_params : keyword arguments, optional
        Parameters to the :class:`CASTable` constructor

    Returns
    -------
    :class:`ImageEmbeddingTable`

    '''

    image_file_list = None
    image_file_list_with_labels = None
    embedding_model_type = None

    def __init__(self, name, **table_params):
        CASTable.__init__(self, name, **table_params)
        self.patch_level = 0

    @classmethod
    def load_files(cls, conn, path, casout=None, columns=None, caslib=None,
                   embedding_model_type='Siamese', n_samples=512,
                   label_level=-2, resize_width=None, resize_height=None):

        '''

        Create ImageEmbeddingTable from files in `path`

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The path to the image directory on the server.
            Path may be absolute, or relative to caslib root if specified.
        casout : dict, optional
            The output table specifications
        columns : list of str, optional
            Specifies the extra columns in the image table.
        caslib : string, optional
            The name of the caslib containing the images.
        embedding_model_type : string, optional
            Specifies the embedding model type that the created table will be applied for training.
            Valid values: Siamese, Triplet, and Quartet.
            Default: Siamese
        n_samples : int, optional
            Number of samples to generate.
            Default: 512
        label_level : int, optional
            Specifies which path level should be used to generate the class labels for each image.
            This class label determines whether a given image pair belongs to the same class.
            For instance, label_level = 1 means the first directory and label_level = -2 means the last directory.
            This internally use the SAS scan function
            (check https://www.sascrunch.com/scan-function.html for more details).
            Default: -2
        resize_width : int, optional
            Specifies the image width that needs be resized to. When resize_width is not given, it will be reset to
            the specified resize_height.
        resize_height : int, optional
            Specifies the image height that needs be resized to. When resize_height is not given, it will be reset to
            the specified resize_width.

        Returns
        -------

        :class:`ImageEmbeddingTable`

        '''

        conn.loadactionset('image', _messagelevel='error')
        conn.loadactionset('sampling', _messagelevel='error')
        conn.loadactionset('deepLearn', _messagelevel='error')

        # check resize options
        if resize_width is not None and resize_height is None:
            resize_height = resize_width
        if resize_width is None and resize_height is not None:
            resize_width = resize_height

        # ignore the unreasonable values for resize
        if resize_width is not None and resize_width <= 0:
            resize_width = None
            resize_height = None

        if resize_height is not None and resize_height <= 0:
            resize_width = None
            resize_height = None

        # if not file_exist_on_server(conn, path):
        #    raise DLPyError('{} does not exist on the server.'.format(path))

        if embedding_model_type.lower() not in ['siamese', 'triplet', 'quartet']:
            raise DLPyError('Only Siamese, Triplet, and Quartet are valid.')

        if casout is None:
            casout = dict(name=random_name())
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if 'name' not in casout:
            casout['name'] = random_name()

        cls.embedding_model_type = embedding_model_type

        with caslibify_context(conn, path, task='load') as (caslib_created, path_created):

            if caslib is None:
                caslib = caslib_created
                path = path_created

            if caslib is None and path is None:
                print('Cannot create a caslib for the provided path. Please make sure that the path is accessible from'
                      'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')

            # load the path information for all the files
            if cls.image_file_list is None:
                castable_with_file_list = dict(name=random_name())
                conn.retrieve('table.loadTable', _messagelevel='error',
                              casout=castable_with_file_list,
                              importOptions=dict(fileType='image', contents=False, recurse=True),
                              path=path_created, caslib=caslib)
                # download all the file information to a dataframe
                n_obs = conn.retrieve('simple.numRows', _messagelevel='error',
                                      table=castable_with_file_list)

                res_fetch = conn.retrieve('table.fetch', _messagelevel='error', maxRows=n_obs['numrows']+100,
                                          fetchVars=['_path_'], to=n_obs['numrows'], table=castable_with_file_list)

                # this stores the entire file path information
                cls.image_file_list = res_fetch['Fetch']['_path_']
                # generate the list using labels as keys
                cls.image_file_list_with_labels = {}
                for file in cls.image_file_list:
                    label = os.path.normpath(file).split(os.sep)[label_level]
                    if label in cls.image_file_list_with_labels:
                        cls.image_file_list_with_labels[label] = \
                            cls.image_file_list_with_labels[label].append(pd.Series([file]))
                    else:
                        cls.image_file_list_with_labels[label] = pd.Series([file])

                conn.retrieve('table.dropTable', _messagelevel='error',
                              table=castable_with_file_list['name'])

                # check whether the image file contains the correct labels
                if len(cls.image_file_list_with_labels) == 1:
                    raise DLPyError('Only one class {} is present in the image files. This could be caused by '
                                    'the wrong labels generated by label_level or the data '
                                    'is highly imbalanced.'.format(cls.image_file_list_with_labels))

            # randomly select n_sample files
            # anchor file list
            # do stratified sampling
            anchor_file_list = []
            for label in cls.image_file_list_with_labels:
                n_samples_per_label = \
                    round(n_samples * (len(cls.image_file_list_with_labels[label])/len(cls.image_file_list)))
                if n_samples_per_label == 0:
                    n_samples_per_label = 1
                sample_list = cls.image_file_list_with_labels[label].sample(n=n_samples_per_label, replace=True)
                sample_list = sample_list.tolist()
                anchor_file_list += sample_list

            anchor_images_casout = cls.__load_images_with_the_file_list(conn, path, caslib, anchor_file_list,
                                                                        resize_width, resize_height)

            siamese_images_casout = None
            positive_images_casout = None
            negative_images_casout = None
            negative_images_casout1 = None

            # Siamese
            if embedding_model_type.lower() == 'siamese':
                siamese_file_list = cls.image_file_list.sample(n=n_samples, replace=True)
                siamese_images_casout = cls.__load_images_with_the_file_list(conn, path, caslib, siamese_file_list,
                                                                             resize_width, resize_height)
                cls.__alter_image_columns(conn, siamese_images_casout, 1)
            # triplet
            elif embedding_model_type.lower() == 'triplet' or embedding_model_type.lower() == 'quartet':
                positive_file_list = pd.Series([])
                negative_file_list = pd.Series([])
                negative_file_list1 = pd.Series([])
                for anchor_file in anchor_file_list:
                    # grab labels
                    temp_path = os.path.normpath(anchor_file)
                    path_tokens = temp_path.split(os.sep)
                    anchor_label = path_tokens[label_level]

                    # generate positive examples
                    positive_file_list = positive_file_list.append(
                        cls.image_file_list_with_labels[anchor_label].sample(n=1, replace=True))
                    # generate negative examples
                    negative_label = anchor_label
                    while negative_label == anchor_label:
                        negative_label = random.choice(list(cls.image_file_list_with_labels.keys()))
                    negative_file_list = negative_file_list.append(
                        cls.image_file_list_with_labels[negative_label].sample(n=1, replace=True))

                    # generate another negative examples with quartet
                    if embedding_model_type.lower() == 'quartet':
                        negative_file_list1 = negative_file_list1.append(
                            cls.image_file_list_with_labels[negative_label].sample(n=1, replace=True))

                # load files into a CAS table
                positive_images_casout = cls.__load_images_with_the_file_list(conn, path, caslib, positive_file_list,
                                                                              resize_width, resize_height)
                cls.__alter_image_columns(conn, positive_images_casout, 1)
                negative_images_casout = cls.__load_images_with_the_file_list(conn, path, caslib, negative_file_list,
                                                                              resize_width, resize_height)
                cls.__alter_image_columns(conn, negative_images_casout, 2)
                if embedding_model_type.lower() == 'quartet':
                    negative_images_casout1 = cls.__load_images_with_the_file_list(conn, path, caslib,
                                                                                   negative_file_list1,
                                                                                   resize_width, resize_height)
                    cls.__alter_image_columns(conn, negative_images_casout1, 3)

            # call dlJoin to generate the final data table
            if embedding_model_type.lower() == 'siamese':
                conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                              casout=casout,
                              joinType='inner', id='_id_',
                              left=anchor_images_casout,
                              right=siamese_images_casout
                              )
            elif embedding_model_type.lower() == 'triplet' or embedding_model_type.lower() == 'quartet':
                conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                              casout=casout,
                              joinType='inner', id='_id_',
                              left=anchor_images_casout,
                              right=positive_images_casout
                              )
                conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                              casout=dict(replace=True, **casout),
                              joinType='inner', id='_id_',
                              left=casout,
                              right=negative_images_casout
                              )
                if embedding_model_type.lower() == 'quartet':
                    conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                                  casout=dict(replace=True, **casout),
                                  joinType='inner', id='_id_',
                                  left=casout,
                                  right=negative_images_casout1
                                  )

            # drop temp tables
            if anchor_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=anchor_images_casout['name'])
            if siamese_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=siamese_images_casout['name'])
            if positive_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=positive_images_casout['name'])
            if negative_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=negative_images_casout['name'])
            if negative_images_casout1:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=negative_images_casout1['name'])

        # generate the final casout table
        code = cls.__generate_codes_for_label(conn, 0, label_level)
        if embedding_model_type.lower() == 'siamese':
            # generate the dissimilar column
            code = code + cls.__generate_codes_for_label(conn, 1, label_level)
            code = code + '_dissimilar_ = 1;'
            code = code + ' if (_label_ = _label_1) then _dissimilar_=0;'
            conn.retrieve('table.partition', _messagelevel='error',
                          table=dict(computedvars=['_fName_', '_label_', '_fName_1', '_label_1', '_dissimilar_'],
                                     computedvarsprogram=code,
                                     name=casout['name']),
                          casout=dict(replace=True, blocksize=32, **casout))
        elif embedding_model_type.lower() == 'triplet':
            # generate other labels
            code = code + cls.__generate_codes_for_label(conn, 1, label_level)
            code = code + cls.__generate_codes_for_label(conn, 2, label_level)
            conn.retrieve('table.partition', _messagelevel='error',
                          table=dict(computedvars=['_fName_', '_label_', '_fName_1', '_label_1',
                                                   '_fName_2', '_label_2'],
                                     computedvarsprogram=code,
                                     name=casout['name']),
                          casout=dict(replace=True, blocksize=32, **casout))
        elif embedding_model_type.lower() == 'quartet':
            # generate other labels
            code = code + cls.__generate_codes_for_label(conn, 1, label_level)
            code = code + cls.__generate_codes_for_label(conn, 2, label_level)
            code = code + cls.__generate_codes_for_label(conn, 3, label_level)
            conn.retrieve('table.partition', _messagelevel='error',
                          table=dict(computedvars=['_fName_', '_label_', '_fName_1', '_label_1',
                                                   '_fName_2', '_label_2', '_fName_3', '_label_3'],
                                     computedvarsprogram=code,
                                     name=casout['name']),
                          casout=dict(replace=True, blocksize=32, **casout))

        out = cls(**casout)
        out.set_connection(conn)

        return out

    # private function
    @staticmethod
    def __alter_image_columns(conn, table, col_index):
        caslib = None
        if 'caslib' in table.keys():
            caslib = table['caslib']
        conn.retrieve('table.alterTable', _messagelevel='error',
                      caslib=caslib,
                      name=table['name'],
                      columns=[dict(name='_image_', rename='_image_' + str(col_index)),
                               dict(name='_path_', rename='_path_' + str(col_index)),
                               dict(name='_type_', rename='_type_' + str(col_index)),
                               dict(name='_size_', rename='_size_' + str(col_index))
                               ]
                      )

    @staticmethod
    def __load_images_with_the_file_list(conn, path, caslib, file_list, resize_width, resize_height):

        # change the message level
        # upload_frame somehow does not honor _messagelevel
        current_msg_level = conn.getSessOpt(name='messagelevel')
        conn.setSessOpt(messageLevel='ERROR')

        # upload file_list
        file_list_casout = dict(name=random_name())

        # use relative path with respect to caslib
        file_list_relative_path = pd.Series()
        i_tot = 0
        for index, value in enumerate(file_list):
            pos = value.find(path)
            # file_list_relative_path = file_list_relative_path.set_value(i_tot, value[pos:])
            file_list_relative_path.at[i_tot] = value[pos:]
            i_tot = i_tot + 1
        conn.upload_frame(file_list_relative_path.to_frame(), casout=file_list_casout, _messagelevel='error')

        # save the file list
        conn.retrieve('table.save', _messagelevel='error',
                      table=file_list_casout,
                      name=file_list_casout['name'] + '.csv', caslib=caslib)

        conn.retrieve('table.dropTable', _messagelevel='error',
                      table=file_list_casout['name'])

        # load images based on the csv file
        # we need use the absolute path here since .csv stores the absolute file location
        images_casout = dict(name=random_name())
        caslib_info = conn.retrieve('caslibinfo', _messagelevel='error',
                                    caslib=caslib)

        # full_path = caslib_info['CASLibInfo']['Path'][0] + file_list_casout['name'] + '.csv'
        # relative to caslib
        csv_path = file_list_casout['name'] + '.csv'

        # use relative paths
        conn.retrieve('image.loadimages', _messagelevel='error',
                      casout=images_casout,
                      caslib=caslib,
                      recurse=True,
                      path=csv_path,
                      pathIsList=True)

        # remove the csv file
        conn.deleteSource(source=file_list_casout['name'] + '.csv', caslib=caslib, _messagelevel='error')

        # resize when it is specified
        if resize_width is not None:
            conn.retrieve('image.processImages', _messagelevel='error',
                          imagefunctions=[
                              {'options': {
                                  'functiontype': 'RESIZE',
                                  'width': resize_width,
                                  'height': resize_height
                              }}
                          ],
                          casout=dict(name=images_casout['name'], replace=True),
                          table=images_casout)

        # reset msg level
        conn.setSessOpt(messageLevel=current_msg_level['messageLevel'])

        return images_casout

    @staticmethod
    def __generate_codes_for_label(conn, col_index, label_level):
        fs = get_server_path_sep(conn)

        if col_index > 0:
            scode = "i{}=find(_path_{},'{}',-length(_path_{})); ".format(col_index, col_index, fs, col_index)
            scode += "length _fName_{} varchar(*); length _label_{} varchar(*); ".format(col_index, col_index)
            scode += "_fName_{}=substr(_path_{}, i+length('{}'), length(_path_{})-i);".format(col_index, col_index,
                                                                                              fs, col_index)
            scode += "_label_{}=scan(_path_{},{},'{}');".format(col_index, col_index, label_level, fs)
        else:
            scode = "i=find(_path_,'{0}',-length(_path_)); ".format(fs)
            scode += "length _fName_ varchar(*); length _label_ varchar(*); "
            scode += "_fName_=substr(_path_, i+length('{0}'), length(_path_)-i);".format(fs)
            scode += "_label_=scan(_path_,{},'{}');".format(label_level, fs)
        return scode

    @property
    def label_freq(self):
        '''
        Summarize the distribution of different image pairs in the ImageEmbeddingTable
        Returns
        -------
        ( frequency for the first image label column,
          frequency for the second image label column,
          for Siamese, frequency for the first + second image label columns and
          frequency for the dissimilar column
          for triplet and quartet, frequency for the third image label column,
          for triplet, frequency for the first + second + third image label columns
          for quartet, frequency for the fourth image label column,
          for quartet, frequency for the first + second + third + fourth image label columns
        )
        '''

        if self.embedding_model_type.lower() == 'siamese':
            code = 'length _label_pair_ varchar(*); _label_pair_ = catx(",", _label_, _label_1)'
            out = self._retrieve('simple.freq', table=dict(name=self, computedVars='_label_pair_',
                                                           computedVarsProgram=code),
                                 inputs=['_label_', '_label_1', '_dissimilar_', '_label_pair_'])['Frequency']

            out = out[['Column', 'FmtVar', 'Level', 'Frequency']]

            # label
            label = out.loc[out.Column == '_label_']
            label = label[['FmtVar', 'Level', 'Frequency']]
            label = label.set_index('FmtVar')
            label.index.name = None
            label = label.astype('int64')

            # label1
            label1 = out.loc[out.Column == '_label1_']
            label1 = label1[['FmtVar', 'Level', 'Frequency']]
            label1 = label1.set_index('FmtVar')
            label1.index.name = None
            label1 = label.astype('int64')

            # label_dissimilar
            label_dissimilar = out.loc[out.Column == '_dissimilar_']
            label_dissimilar = label_dissimilar[['FmtVar', 'Level', 'Frequency']]
            label_dissimilar = label_dissimilar.set_index('FmtVar')
            label_dissimilar.index.name = None
            label_dissimilar = label_dissimilar.astype('int64')

            # label_dissimilar
            label_pair = out.loc[out.Column == '_label_pair_']
            label_pair = label_pair[['FmtVar', 'Level', 'Frequency']]
            label_pair = label_pair.set_index('FmtVar')
            label_pair.index.name = None
            label_pair = label_pair.astype('int64')

            return label, label1, label_pair, label_dissimilar

        elif self.embedding_model_type.lower() == 'triplet':
            code = 'length _triplet_ varchar(*); _triplet_ = catx(",", _label_, _label_1, _label_2)'
            out = self._retrieve('simple.freq', table=dict(name=self, computedVars='_triplet_',
                                                           computedVarsProgram=code),
                                 inputs=['_label_', '_label_1', '_label_2', '_triplet_'])['Frequency']

            out = out[['Column', 'FmtVar', 'Level', 'Frequency']]

            # label
            label = out.loc[out.Column == '_label_']
            label = label[['FmtVar', 'Level', 'Frequency']]
            label = label.set_index('FmtVar')
            label.index.name = None
            label = label.astype('int64')

            # label1
            label1 = out.loc[out.Column == '_label1_']
            label1 = label1[['FmtVar', 'Level', 'Frequency']]
            label1 = label1.set_index('FmtVar')
            label1.index.name = None
            label1 = label.astype('int64')

            # label 2
            label2 = out.loc[out.Column == '_label_2']
            label2 = label2[['FmtVar', 'Level', 'Frequency']]
            label2 = label2.set_index('FmtVar')
            label2.index.name = None
            label2 = label2.astype('int64')

            # label_triplet
            label_triplet = out.loc[out.Column == '_triplet_']
            label_triplet = label_triplet[['FmtVar', 'Level', 'Frequency']]
            label_triplet = label_triplet.set_index('FmtVar')
            label_triplet.index.name = None
            label_triplet = label_triplet.astype('int64')

            return label, label1, label2, label_triplet
        elif self.embedding_model_type.lower() == 'quartet':
            code = 'length _quartet_ varchar(*); _quartet_ = catx(",", _label_, _label_1, _label_2, _label_3)'
            out = self._retrieve('simple.freq', table=dict(name=self, computedVars='_quartet_',
                                                           computedVarsProgram=code),
                                 inputs=['_label_', '_label_1', '_label_2', '_label_3', '_quartet_'])['Frequency']

            out = out[['Column', 'FmtVar', 'Level', 'Frequency']]

            # label
            label = out.loc[out.Column == '_label_']
            label = label[['FmtVar', 'Level', 'Frequency']]
            label = label.set_index('FmtVar')
            label.index.name = None
            label = label.astype('int64')

            # label1
            label1 = out.loc[out.Column == '_label1_']
            label1 = label1[['FmtVar', 'Level', 'Frequency']]
            label1 = label1.set_index('FmtVar')
            label1.index.name = None
            label1 = label.astype('int64')

            # label 2
            label2 = out.loc[out.Column == '_label_2']
            label2 = label2[['FmtVar', 'Level', 'Frequency']]
            label2 = label2.set_index('FmtVar')
            label2.index.name = None
            label2 = label2.astype('int64')

            # label 3
            label3 = out.loc[out.Column == '_label_3']
            label3 = label3[['FmtVar', 'Level', 'Frequency']]
            label3 = label3.set_index('FmtVar')
            label3.index.name = None
            label3 = label3.astype('int64')

            # label_quartet
            label_quartet = out.loc[out.Column == '_quartet_']
            label_quartet = label_quartet[['FmtVar', 'Level', 'Frequency']]
            label_quartet = label_quartet.set_index('FmtVar')
            label_quartet.index.name = None
            label_quartet = label_quartet.astype('int64')

            return label, label1, label2, label3, label_quartet

    def show(self, n_image_pairs=5, randomize=False, figsize=None, where=None):

        '''

        Display a grid of images for ImageEmbeddingTable

        Parameters
        ----------
        n_image_pairs : int, optional
            Specifies the number of image pairs to be displayed.
            If nimage is greater than the maximum number of image pairs in the
            table, it will be set to this maximum number.
            Note: Specifying a large value for n_image_pairs can lead to slow performance.
        randomize : bool, optional
            Specifies whether to randomly choose the images for display.
        figsize : int, optional
            Specifies the size of the fig that contains the image.
        where : string, optional
            Specifies the SAS Where clause for selecting images to be shown.
            One example is as follows:
            my_images.show(n_image_pairs=2, where='_id_ eq 57')

        '''

        n_scale = 1
        fetch_vars = []
        if self.embedding_model_type.lower() == 'siamese':
            fetch_vars = ['_id_', '_path_', '_image_1', '_label_1', '_path_1', '_dissimilar_']
            n_scale = 2
        elif self.embedding_model_type.lower() == 'triplet':
            fetch_vars = ['_id_', '_path_', '_image_1', '_label_1', '_path_1', '_image_2', '_label_2', '_path_2']
            n_scale = 3
        elif self.embedding_model_type.lower() == 'quartet':
            fetch_vars = ['_id_', '_path_', '_image_1', '_label_1', '_path_1', '_image_2', '_label_2', '_path_2',
                          '_image_3', '_label_3', '_path_3']
            n_scale = 4

        temp_tbl = self.__fetch_images(self, n_image_pairs, randomize, where, fetch_vars)

        # fix ncol
        ncol = 1

        if n_image_pairs > ncol:
            nrow = n_image_pairs // ncol + 1
        else:
            nrow = 1
            ncol = n_image_pairs
        if figsize is None:
            figsize = (4 * n_scale, 16)
        fig = plt.figure(figsize=figsize)

        for i in range(n_image_pairs):

            # display the anchor image
            image = temp_tbl['Images']['Image'][i]
            if 'Label' in temp_tbl['Images'].columns:
                label = temp_tbl['Images']['Label'][i]
            else:
                label = 'N/A'
            ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 1)
            id_content = temp_tbl['Images']['_id_'][i]

            if self.embedding_model_type.lower() == 'siamese':
                ax.set_title('Label: {}\nPair: {}'.format(label, id_content))
            elif self.embedding_model_type.lower() == 'triplet':
                ax.set_title('Anchor: {}\nTriplet: {}'.format(label, id_content))
            elif self.embedding_model_type.lower() == 'quartet':
                ax.set_title('Anchor: {}\nQuartet: {}'.format(label, id_content))

            if len(image.size) == 2:
                plt.imshow(np.array(image), cmap='Greys_r')
            else:
                plt.imshow(image)
            plt.xticks([]), plt.yticks([])

            if self.embedding_model_type.lower() == 'siamese':
                # display the pair image
                image = temp_tbl['Images']['_image_1'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_1'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 2)
                dissimilar = temp_tbl['Images']['_dissimilar_'][i]
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Label: {}\nPair: {} Dissimilar: {}'.format(label, id_content, dissimilar))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])
            elif self.embedding_model_type.lower() == 'triplet':
                # display the positive image
                image = temp_tbl['Images']['_image_1'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_1'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 2)
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Positive: {}\nTriplet: {}'.format(label, id_content))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])
                # display the negative image
                image = temp_tbl['Images']['_image_2'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_2'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 3)
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Negative: {}\nTriplet: {}'.format(label, id_content))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])
            elif self.embedding_model_type.lower() == 'quartet':
                # display the positive image
                image = temp_tbl['Images']['_image_1'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_1'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 2)
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Positive: {}\nQuartet: {}'.format(label, id_content))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])
                # display the negative image
                image = temp_tbl['Images']['_image_2'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_2'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 3)
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Negative: {}\nQuartet: {}'.format(label, id_content))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])
                # display another negative image
                image = temp_tbl['Images']['_image_3'][i]
                if 'Label' in temp_tbl['Images'].columns:
                    label = temp_tbl['Images']['_label_3'][i]
                else:
                    label = 'N/A'
                ax = fig.add_subplot(nrow, ncol * n_scale, n_scale * i + 4)
                id_content = temp_tbl['Images']['_id_'][i]
                ax.set_title('Negative1: {}\nQuartet: {}'.format(label, id_content))
                if len(image.size) == 2:
                    plt.imshow(np.array(image), cmap='Greys_r')
                else:
                    plt.imshow(image)
                plt.xticks([]), plt.yticks([])

        plt.tight_layout()
        plt.show()

    @staticmethod
    def __fetch_images(self, n_image_pairs=5, randomize=False, where=None, id=None):
        n_image_pairs = min(n_image_pairs, len(self))
        # put where clause to select images
        self.params['where'] = where
        # restrict the number of observations to be shown
        try:
            # we use numrows to check if where clause is valid
            max_obs = self.numrows().numrows
            n_image_pairs = min(max_obs, n_image_pairs)
        except AttributeError:
            self.params['where'] = None
            warn("Where clause doesn't take effect, because encounter an error while processing where clause. "
                 "Please check your where clause.")

        if randomize:
            temp_tbl = self.retrieve('image.fetchimages', _messagelevel='error',
                                     table=dict(
                                         computedvars=['random_index'],
                                         computedvarsprogram='call streaminit(-1);'
                                                             'random_index='
                                                             'rand("UNIFORM");',
                                         **self.to_table_params()),
                                     image='_image_',
                                     sortby='random_index', to=n_image_pairs,
                                     fetchImagesVars=id)
        else:
            temp_tbl = self._retrieve('image.fetchimages', to=n_image_pairs, image='_image_',
                                      fetchImagesVars=id)
        # remove the where clause
        self.params['where'] = None

        return temp_tbl
