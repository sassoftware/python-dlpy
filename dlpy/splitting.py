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

''' Functions to support different splitting schemes '''

from swat.cas.table import CASTable
from .images import ImageTable
from .utils import random_name


def two_way_split(tbl, test_rate=20, stratify=True, im_table=True, stratify_by='_label_',
                  image_col='_image_', train_name=None, test_name=None, **kwargs):
    '''
    Split image data into training and testing sets

    Parameters
    ----------
    tbl : CASTable
        The CAS table to split
    test_rate : double, optional
        Specifies the proportion of the testing data set,
        e.g. 20 mean 20% of the data will be in the testing set.
    stratify : boolean, optional
        If True stratify the sampling by the stratify_by column name
        If False do random sampling without stratification
    im_table : boolean, optional
        If True outputs are converted to an imageTable
        If False CASTables are returned with all columns
    image_col : string
        Name of image column if returning ImageTable
    train_name : string
        Specifies the output table name for the training set
    test_name : string
        Specifies the output table name for the test set
    **kwargs : keyword arguments, optional
        Additional keyword arguments to the `sample.stratified` or
        'sample.src' actions

    Returns
    -------
    ( training CASTable, testing CASTable )

    '''
    if train_name is None:
        train_tbl_name = random_name('train')
    elif isinstance(train_name, str):
        train_tbl_name = train_name
    else:
        raise ValueError('train_name must be a string')

    if test_name is None:
        test_tbl_name = random_name('test')
    elif isinstance(test_name, str):
        test_tbl_name = test_name
    else:
        raise ValueError('test_name must be a string')

    temp_tbl_name = random_name('Temp')

    tbl._retrieve('loadactionset', actionset='sampling')

    partind_name = random_name(name='PartInd_', length=2)

    tbl_columns = tbl.columns.tolist()

    if stratify:
        tbl._retrieve('sampling.stratified',
                      output=dict(casout=temp_tbl_name, copyvars='all',
                                  partindname=partind_name),
                      samppct=test_rate, samppct2=100 - test_rate, partind=True,
                      table=dict(groupby=stratify_by, **tbl.to_table_params()), **kwargs)

    else:
        tbl._retrieve('sampling.srs',
                      output=dict(casout=temp_tbl_name, copyvars='all',
                                  partindname=partind_name),
                      samppct=test_rate, samppct2=100 - test_rate, partind=True,
                      table=dict(**tbl.to_table_params()), **kwargs)

    test = tbl._retrieve('table.partition',
                         table=dict(where='{}=1'.format(partind_name),
                                    name=temp_tbl_name, Vars=tbl_columns),
                         casout=dict(name=test_tbl_name, replace=True,
                                     blocksize=128))['casTable']

    train = tbl._retrieve('table.partition',
                          table=dict(where='{}=2'.format(partind_name),
                                     name=temp_tbl_name, Vars=tbl_columns),
                          casout=dict(name=train_tbl_name, replace=True,
                                      blocksize=128))['casTable']

    tbl._retrieve('table.dropTable',
                  name=temp_tbl_name)

    if im_table:
        train_im = ImageTable.from_table(train, label_col=stratify_by, image_col=image_col,
                                         casout=dict(name=train.name))
        test_im = ImageTable.from_table(test, label_col=stratify_by, image_col=image_col,
                                        casout=dict(name=test.name))
        return train_im, test_im
    else:
        return train, test


def three_way_split(tbl, valid_rate=20, test_rate=20, stratify=True, im_table=True,
                    stratify_by='_label_', image_col='_image_', train_name=None,
                    valid_name=None, test_name=None, **kwargs):
    '''
    Split image data into training and testing sets

    Parameters
    ----------
    tbl : CASTable
        The CAS table to split
    valid_rate : double, optional
        Specifies the proportion of the validation data set,
        e.g. 20 mean 20% of the images will be in the validation set.
    test_rate : double, optional
        Specifies the proportion of the testing data set,
        e.g. 20 mean 20% of the images will be in the testing set.
        Note: the total of valid_rate and test_rate cannot be exceed 100
    stratify : boolean, optional
        If True stratify the sampling by the stratify_by column name
        If False do random sampling without stratification
    im_table : boolean, optional
        If True outputs are converted to an imageTable
        If False CASTables are returned with all columns
    stratify_by : string, optional
        The variable to stratify by
    image_col : string
        Name of image column if returning ImageTable
    train_name : string
        Specifies the output table name for the training set
    valid_name : string
        Specifies the output table name for the validation set
    test_name : string
        Specifies the output table name for the test set
    **kwargs : keyword arguments, optional
        Additional keyword arguments to the `sample.stratified` or
        'sample.srs' actions

    Returns
    -------
    ( train CASTable, valid CASTable, test CASTable )

    '''

    if train_name is None:
        train_tbl_name = random_name('train')
    elif isinstance(train_name, str):
        train_tbl_name = train_name
    else:
        raise ValueError('train_name must be a string')

    if valid_name is None:
        valid_tbl_name = random_name('valid')
    elif isinstance(test_name, str):
        valid_tbl_name = valid_name
    else:
        raise ValueError('test_name must be a string')

    if test_name is None:
        test_tbl_name = random_name('test')
    elif isinstance(test_name, str):
        test_tbl_name = test_name
    else:
        raise ValueError('test_name must be a string')

    temp_tbl_name = random_name('Temp')

    tbl._retrieve('loadactionset', actionset='sampling')

    partind_name = random_name(name='part_ind_', length=2)
    tbl_columns = tbl.columns.tolist()

    if stratify:
        tbl._retrieve('sampling.stratified',
                      output=dict(casout=temp_tbl_name, copyvars='all',
                                  partindname=partind_name),
                      samppct=valid_rate, samppct2=test_rate,
                      partind=True,
                      table=dict(groupby=stratify_by, **tbl.to_table_params()), **kwargs)
    else:
        tbl._retrieve('sampling.srs',
                      output=dict(casout=temp_tbl_name, copyvars='all',
                                  partindname=partind_name),
                      samppct=valid_rate, samppct2=test_rate, partind=True,
                      table=dict(**tbl.to_table_params()), **kwargs)



    train = tbl._retrieve('table.partition',
                          table=dict(where='{}=0'.format(partind_name),
                                     name=temp_tbl_name, Vars=tbl_columns),
                          casout=dict(name=train_tbl_name, replace=True))['casTable']

    valid = tbl._retrieve('table.partition',
                          table=dict(where='{}=1'.format(partind_name),
                                     name=temp_tbl_name, Vars=tbl_columns),
                          casout=dict(name=valid_tbl_name, replace=True))['casTable']

    test = tbl._retrieve('table.partition',
                         table=dict(where='{}=2'.format(partind_name),
                                    name=temp_tbl_name, Vars=tbl_columns),
                         casout=dict(name=test_tbl_name, replace=True))['casTable']

    tbl._retrieve('table.dropTable',
                  name=temp_tbl_name)
    if im_table:
        train_im = ImageTable.from_table(train, label_col=stratify_by, image_col=image_col,
                                         casout=dict(name=train.name))
        valid_im = ImageTable.from_table(valid, label_col=stratify_by, image_col=image_col,
                                        casout=dict(name=valid.name))
        test_im = ImageTable.from_table(test, label_col=stratify_by, image_col=image_col,
                                        casout=dict(name=test.name))

        return train_im, valid_im, test_im
    else:
        return train, valid, test
