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

import matplotlib.pyplot as plt
import numpy as np
from swat.cas.table import CASTable

from dlpy import ImageTable
from .utils import random_name, image_blocksize, caslibify_context, get_server_path_sep, get_cas_host_type
from warnings import warn


class ImageEmbeddingTable(ImageTable):
    '''

    Specialized CASTable for Image Embedding Data

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

    castable_with_file_list = None

    @classmethod
    def load_files(cls, conn, path, casout=None, columns=None, caslib=None,
                   embedding_model_type='Siamese', n_samples=512):

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
            Three model types are supported: Siamese, Triplet, and Quartet.
            Default: Siamese
        n_samples : int, None by default
            Number of samples to generate.
            Default: 512

        Returns
        -------

        :class:`ImageEmbeddingTable`

        '''

        conn.loadactionset('image', _messagelevel='error')
        conn.loadactionset('sampling', _messagelevel='error')
        conn.loadactionset('deepLearn', _messagelevel='error')

        if casout is None:
            casout = dict(name=random_name())
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if 'name' not in casout:
            casout['name'] = random_name()

        # only load the file list once
        load_file_list = False
        if cls.castable_with_file_list is None:
            cls.castable_with_file_list = dict(name=random_name())
            load_file_list = True

        with caslibify_context(conn, path, task='load') as (caslib_created, path_created):

            if caslib is None:
                caslib = caslib_created
                path = path_created

            if caslib is None and path is None:
                print('Cannot create a caslib for the provided path. Please make sure that the path is accessible from'
                      'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')

            # load the path information for all the files
            if load_file_list:
                conn.retrieve('table.loadTable', _messagelevel='error',
                              casout=cls.castable_with_file_list,
                              importOptions=dict(fileType='image', contents=False, recurse=True),
                              path=path_created, caslib=caslib)

            # randomly select n_sample files
            # anchor file list
            anchor_casout = dict(name=random_name())
            conn.retrieve('sampling.srs', _messagelevel='error',
                          table=cls.castable_with_file_list,
                          output=dict(casout=anchor_casout, copyVars='_path_'),
                          fixedObs=n_samples
                          )

            conn.retrieve('table.save', _messagelevel='error',
                          table=anchor_casout,
                          name=anchor_casout['name']+'.csv', caslib=caslib)
            # Siamese
            siamese_casout = None
            if embedding_model_type.lower() == 'siamese':
                siamese_casout = dict(name=random_name())
                conn.retrieve('sampling.srs', _messagelevel='error',
                              table=cls.castable_with_file_list,
                              output=dict(casout=siamese_casout, copyVars='_path_'),
                              fixedObs=n_samples
                              )
                conn.retrieve('table.save', _messagelevel='error',
                              table=siamese_casout,
                              name=siamese_casout['name'] + '.csv', caslib=caslib)

            # call loadImages to load the selected files
            # need use the absolute path since loadTable only geenerates this kind
            # so no caslib is used
            anchor_images_casout = dict(name=random_name())
            conn.retrieve('image.loadimages', _messagelevel='error',
                          casout=anchor_images_casout,
                          recurse=True, labellevels=-1,
                          path=anchor_casout['name']+'.csv',
                          pathIsList=True)
            if siamese_casout:
                siamese_images_casout = dict(name=random_name())
                conn.retrieve('image.loadimages', _messagelevel='error',
                              casout=siamese_images_casout,
                              recurse=True, labellevels=-1,
                              path=siamese_casout['name'] + '.csv',
                              pathIsList=True)
                cls.__alter_image_columns(conn, siamese_images_casout, 1)

            # call dlJoin to generate the final data table
            temp_casout = dict(name=random_name())
            if embedding_model_type.lower() == 'siamese':
                conn.retrieve('deepLearn.dlJoin', _messagelevel='error',
                              casout=temp_casout,
                              joinType='inner', id='_id_',
                              left=anchor_images_casout,
                              right=siamese_images_casout
                              )

            # drop temp tables
            if anchor_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=anchor_casout['name'])
            if anchor_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=anchor_images_casout['name'])

            if siamese_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=siamese_casout['name'])
            if siamese_images_casout:
                conn.retrieve('table.dropTable', _messagelevel='error',
                              name=siamese_images_casout['name'])

        # generate the final casout table
        sep_ = get_server_path_sep(conn)
        if embedding_model_type.lower() == 'siamese':
            # generate the dissimilar column
            code = '_dissimilar_ = 0;'
            code = code + ' if (_label_ = label_1) then _dissimilar_=1;'
            conn.retrieve('table.partition', _messagelevel='error',
                          table=dict(computedvars=['_dissimilar_'],
                                     computedvarsprogram=code,
                                     name=temp_casout['name']),
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
                      columns=[dict(name='_image_', rename='_image_'+str(col_index)),
                               dict(name='_path_', rename='_path_' + str(col_index)),
                               dict(name='_type_', rename='_type_' + str(col_index)),
                               dict(name='_label_', rename='_label_' + str(col_index))
                               ]
                      )

