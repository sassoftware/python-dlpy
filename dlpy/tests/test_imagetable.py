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

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.

import os
import unittest
import swat
import swat.utils.testing as tm
from dlpy.images import ImageTable
from dlpy import Sequential
from dlpy.model import *
from dlpy.layers import *
from dlpy.splitting import two_way_split
from dlpy.applications import *


class TestImageTable(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.s = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.s)

        cls.server_sep = '\\'
        if cls.server_type.startswith("lin") or cls.server_type.startswith("osx"):
            cls.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            cls.data_dir = os.environ.get('DLPY_DATA_DIR')
            if cls.data_dir.endswith(cls.server_sep):
                cls.data_dir = cls.data_dir[:-1]
            cls.data_dir += cls.server_sep

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_two_way_split(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        tr_img, te_img = two_way_split(my_images, test_rate = 20, seed = 123)
        self.assertTrue(tr_img.numrows().numrows == 328)
        self.assertTrue(tr_img.cls_cols == '_label_')
        self.assertTrue(tr_img.id_col == '_id_')
        self.assertTrue(tr_img.filename_col == '_filename_')

    def test_two_way_split2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file',
                                     cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        self.assertTrue(tr_img.image_cols == 'img')
        self.assertTrue(te_img.cls_cols == 'label')
        self.assertTrue(te_img.id_col == 'id')
        self.assertTrue(tr_img.filename_col == 'file')

        tr_img, te_img = two_way_split(test, test_rate = 20, stratify = False, stratify_by = test.cls_cols, seed = 123)
        self.assertTrue(tr_img.image_cols == 'img')
        self.assertTrue(te_img.cls_cols == 'label')
        self.assertTrue(te_img.id_col == 'id')
        self.assertTrue(tr_img.filename_col == 'file')

    def test_as_patches(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        tr_img, te_img = two_way_split(my_images, test_rate = 20, seed = 123)
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        te_img.random_mutations(darken = True)

    def test_as_random_patches(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        tr_img, te_img = two_way_split(my_images, test_rate = 20, seed = 123)
        te_img.as_random_patches()
        te_img.random_mutations(darken = True)

    def test_from_table(self):
        self.s.table.addcaslib(activeonadd = False, datasource = {'srctype': 'path'}, name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/', subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img',
                                     id_col = 'id', filename_col = 'file', cls_cols = 'label')
        channel_mean = [round(i, 2) for i in test.channel_means]
        self.assertTrue(channel_mean[0] == 127.28)
        self.assertTrue(test.image_cols == 'img')
        self.assertTrue(test.cls_cols == 'label')
        self.assertTrue(test.id_col == 'id')
        self.assertTrue(test.filename_col == 'file')

    def test_from_table2(self):
        self.s.table.addcaslib(activeonadd = False,
                              datasource = {'srctype': 'path'},
                              name = 'dnfs',
                              path = '/bigdisk/lax/dlpy/',
                              subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        with self.assertRaises(ValueError):
            ImageTable.from_table(tbl = self.s.CASTable('data'), id_col = 'id', filename_col = 'file',
                                  cls_cols = 'label')
        with self.assertRaises(ValueError):
            test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                         filename_col = 'file', cls_cols = 'labels')


    def test_load_images(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        my_images = ImageTable.load_files(self.s, path=img_path)
        self.assertTrue(len(my_images) > 0)

    def test_resize(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.resize(width=200, height=200, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        self.assertTrue(int(out[1]) == 200)
        self.assertTrue(int(out[2]) == 200)
        self.assertTrue(int(out[3]) == 200)
        self.assertTrue(int(out[4]) == 200)
        self.assertTrue(int(out[5]) == 200)
        self.assertTrue(int(out[6]) == 200)

    def test_crop(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.crop(x=0, y=0, width=200, height=200, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        self.assertTrue(int(out[1]) == 200)
        self.assertTrue(int(out[2]) == 200)
        self.assertTrue(int(out[3]) == 200)
        self.assertTrue(int(out[4]) == 200)
        self.assertTrue(int(out[5]) == 200)
        self.assertTrue(int(out[6]) == 200)

    def test_as_patches(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.as_patches(x=0, y=0, width=200, height=200, step_size=200,
                                   output_width=100, output_height=100, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        self.assertTrue(int(out[1]) == 100)
        self.assertTrue(int(out[2]) == 100)
        self.assertTrue(int(out[3]) == 100)
        self.assertTrue(int(out[4]) == 100)
        self.assertTrue(int(out[5]) == 100)
        self.assertTrue(int(out[6]) == 100)
