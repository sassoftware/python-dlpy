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

        out = my_images.as_patches(x=0, y=0, width=None, height=None, step_size=None,
                                   output_width=None, output_height=None, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        self.assertTrue(int(out[1]) == 224)
        self.assertTrue(int(out[2]) == 224)
        self.assertTrue(int(out[3]) == 224)
        self.assertTrue(int(out[4]) == 224)
        self.assertTrue(int(out[5]) == 224)
        self.assertTrue(int(out[6]) == 224)

        out = my_images.as_patches(x=0, y=0, width=None, height=200, step_size=None,
                                   output_width=None, output_height=None, inplace=False)
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



    def test_as_random_patches(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.as_random_patches(x=0, y=0, width=200, height=200, step_size=200,
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

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.as_random_patches(x=0, y=0, width=None, height=None, step_size=None,
                                   output_width=None, output_height=None, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        self.assertTrue(int(out[1]) == 224)
        self.assertTrue(int(out[2]) == 224)
        self.assertTrue(int(out[3]) == 224)
        self.assertTrue(int(out[4]) == 224)
        self.assertTrue(int(out[5]) == 224)
        self.assertTrue(int(out[6]) == 224)

        my_images = ImageTable.load_files(self.s, path=self.data_dir+'giraffe_dolphin_small')
        out = my_images.as_random_patches(x=0, y=0, width=None, height=200, step_size=None,
                                   output_width=None, output_height=None, inplace=False)
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
