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


def two_way_split(tbl, test_rate=20, stratify_by='_label_',
                  image_col='_image_', **kwargs):
    '''
    Split image data into training and testing sets

    Parameters
    ----------
    tbl : CASTable
        The CAS table to split
    test_rate : double, optional
        Specifies the proportion of the testing data set,
        e.g. 20 mean 20% of the data will be in the testing set.
    stratify_by : string, optional
        The variable to stratify by
    **kwargs : keyword arguments, optional
        Additional keyword arguments to the `sample.stratified` action

    Returns
    -------
    ( training CASTable, testing CASTable )

    '''
    train_tbl_name = random_name()
    test_tbl_name = random_name()
    temp_tbl_name = random_name('Temp')

    tbl._retrieve('loadactionset', actionset='sampling')

    partindname = random_name(name='PartInd_', length=2)

    tbl._retrieve('sampling.stratified',
                  output=dict(casout=temp_tbl_name, copyvars='all',
                              partindname=partindname),
                  samppct=test_rate, samppct2=100 - test_rate, partind=True,
                  table=dict(groupby=stratify_by, **tbl.to_table_params()), **kwargs)

    train = tbl._retrieve('table.partition',
                          table=dict(where='{}=2'.format(partindname),
                                     name=temp_tbl_name),
                          casout=dict(name=train_tbl_name, replace=True,
                                      blocksize=128))['casTable']

    test = tbl._retrieve('table.partition',
                         table=dict(where='{}=1'.format(partindname),
                                    name=temp_tbl_name),
                         casout=dict(name=test_tbl_name, replace=True,
                                     blocksize=128))['casTable']

    tbl._retrieve('table.dropTable',
                  name=temp_tbl_name)

    return (ImageTable.from_table(train, label_col=stratify_by, image_col=image_col),
            ImageTable.from_table(test, label_col=stratify_by, image_col=image_col))


def three_way_split(tbl, valid_rate=20, test_rate=20, stratify_by='_label_', **kwargs):
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
    stratify_by : string, optional
        The variable to stratify by
    **kwargs : keyword arguments, optional
        Additional keyword arguments to the `sample.stratified` action

    Returns
    -------
    ( train CASTable, valid CASTable, test CASTable )

    '''
    train_tbl_name = random_name()
    valid_tbl_name = random_name()
    test_tbl_name = random_name()
    temp_tbl_name = random_name('Temp')

    tbl._retrieve('loadactionset', actionset='sampling')

    partindname = random_name(name='PartInd_', length=2)

    tbl._retrieve('sampling.stratified',
                  output=dict(casout=temp_tbl_name, copyvars='all',
                              partindname=partindname),
                  samppct=valid_rate, samppct2=test_rate,
                  partind=True,
                  table=dict(groupby=stratify_by, **tbl.to_table_params()), **kwargs)

    train = tbl._retrieve('table.partition',
                          table=dict(where='{}=0'.format(partindname),
                                     name=temp_tbl_name),
                          casout=train_tbl_name)['casTable']

    valid = tbl._retrieve('table.partition',
                          table=dict(where='{}=1'.format(partindname),
                                     name=temp_tbl_name),
                          casout=valid_tbl_name)['casTable']

    test = tbl._retrieve('table.partition',
                         table=dict(where='{}=2'.format(partindname),
                                    name=temp_tbl_name),
                         casout=test_tbl_name)['casTable']

    tbl._retrieve('table.dropTable',
                  name=temp_tbl_name)

    return (ImageTable.from_table(train),
            ImageTable.from_table(valid),
            ImageTable.from_table(test))
