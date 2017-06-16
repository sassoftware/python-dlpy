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


def image_generator(sess, img_path, tr_te_split=True, test_rate=20):
    '''
    Function to generate CAS table including the image data as "_image_" and labels as "_label".

    Parameters:

    ----------
    sess :
        Specify the session of the CAS connection.
    img_path : string
        Specify the path of the image data.
        Note: images should be save in the form of parent_dir/label/image_files.
    tr_te_split : boolean, optional.
        Specify whether to split the data set in to training and testing set.
        Default : True
    test_rate : double, between 0 and 100, optional.
        Specify the proportion of the testing data set, e.g. 20 mean 20% of the images will be in the testing set.
        Default : 20.

    Returns
    -------
    If tr_te_split is True, return two CAS tables, trainTbl and validTbl.
    If tr_te_split is False, return s CAS table, dataTbl.

    '''
    if not sess.queryactionset('image')['image']:
        sess.loadactionset('image')
    sess.image.loadimages(
        casout=dict(name='temp_tbl', replace=True, blocksize=128),
        distribution=dict(type='random'), recurse=True, labelLevels=-1,
        path=img_path)

    if tr_te_split:
        if not sess.queryactionset('sampling')['sampling']:
            sess.loadactionset('sampling')
        sess.stratified(output=dict(casOut=dict(name="data", replace=True), copyVars="ALL"),
                        samppct=test_rate, samppct2=100 - test_rate,
                        table=dict(name='temp_tbl', groupby='_label_'))
        validTbl = sess.CASTable('data', where='_partind_=1')
        trainTbl = sess.CASTable('data', where='_partind_=2')
        return trainTbl, validTbl
    else:
        dataTbl = sess.CASTable('temp_tbl')
        return dataTbl
