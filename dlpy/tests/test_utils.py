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


import unittest
import swat
import swat.utils.testing as tm
from dlpy.utils import *


class TestUtils(unittest.TestCase):
    '''
    If you are using dlpy on a Windows machine, then copy datasources/dlpy_obj_det_test to both DLPY_DATA_DIR and
    DLPY_DATA_DIR_LOCAL. If you are using dlpy on a linux machine, then copy datasources/dlpy_obj_det_test to
    DLPY_DATA_DIR only.
    '''
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None
    data_dir_local = None

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

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            cls.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if cls.data_dir_local.endswith(cls.server_sep):
                cls.data_dir_local = cls.data_dir_local[:-1]
            cls.data_dir_local += cls.server_sep

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_camelcase_to_underscore(self):
        underscore = camelcase_to_underscore('includeBias')
        self.assertTrue('include_bias' == underscore)

    def test_camelcase_to_underscore(self):
        underscore = underscore_to_camelcase('include_bias')
        self.assertTrue('includeBias' == underscore)

    def test_create_object_detection_table(self):

        if platform.system().startswith('Win'):
            if self.data_dir is None or self.data_dir_local is None:
                unittest.TestCase.skipTest(self, "DLPY_DATA_DIR or DLPY_DATA_DIR_LOCAL is not set in "
                                                 "the environment variables")
            create_object_detection_table(self.s, data_path=self.data_dir+'dlpy_obj_det_test',
                                          local_path=self.data_dir_local+'dlpy_obj_det_test',
                                          coord_type='yolo',
                                          output='output')
            create_object_detection_table(self.s, data_path=self.data_dir+'dlpy_obj_det_test',
                                          local_path=self.data_dir_local+'dlpy_obj_det_test',
                                          coord_type='coco',
                                          output='output')
        else:
            if self.data_dir is None:
                unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

            create_object_detection_table(self.s, coord_type='yolo', output='output',
                                          data_path=self.data_dir+'dlpy_obj_det_test')

            create_object_detection_table(self.s, data_path=self.data_dir+'dlpy_obj_det_test',
                                          coord_type='coco', output='output')

    def test_get_anchors(self):
        if platform.system().startswith('Win'):
            if self.data_dir is None or self.data_dir_local is None:
                unittest.TestCase.skipTest(self, "DLPY_DATA_DIR or DLPY_DATA_DIR_LOCAL is not set in "
                                                 "the environment variables")

            create_object_detection_table(self.s, data_path=self.data_dir+'dlpy_obj_det_test',
                                          local_path=self.data_dir_local+'dlpy_obj_det_test',
                                          coord_type='yolo',
                                          output='output')
        else:
            if self.data_dir is None:
                unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

            create_object_detection_table(self.s, coord_type='yolo', output='output',
                                          data_path=self.data_dir+'dlpy_obj_det_test')

        get_anchors(self.s, coord_type='yolo', data='output')
