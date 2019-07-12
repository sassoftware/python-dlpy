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
from dlpy.images import ImageTable


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
            if cls.data_dir_local.startswith('/'):
                sep_ = '/'
            else:
                sep_ = '\\'
            if cls.data_dir_local.endswith(sep_):
                cls.data_dir_local = cls.data_dir_local[:-1]
            cls.data_dir_local += sep_

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

    def test_underscore_to_camelcase(self):
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
        # there are 11 images where all contains 3 instance.
        # If annotation files are parsed correctly, _nObjects_ column is 3 for all records.
        a = self.s.CASTable('output')
        self.assertTrue(self.s.fetch('output', fetchvars='_nObjects_').Fetch['_nObjects_'].tolist() == [3.0]*len(a))

    def test_create_object_detection_table_2(self):
        # make sure that txt files are already in self.data_dir + 'dlpy_obj_det_test', otherwise the test will fail.
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        create_object_detection_table(self.s, data_path = self.data_dir + 'dlpy_obj_det_test',
                                      coord_type = 'yolo',
                                      output = 'output')
        # there are 11 images where all contains 3 instance.
        # If annotation files are parsed correctly, _nObjects_ column is 3 for all records.
        a = self.s.CASTable('output')
        self.assertTrue(self.s.fetch('output', fetchvars='_nObjects_').Fetch['_nObjects_'].tolist() == [3.0]*len(a))

    def test_create_object_detection_table_3(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        # If coord_type is not either 'yolo' or 'coco', an error should be thrown
        self.assertRaises(ValueError, lambda:create_object_detection_table(self.s, 
                                      data_path = self.data_dir + 'dlpy_obj_det_test',
                                      coord_type = 'invalid_val',
                                      output = 'output'))

    def test_create_object_detection_table_non_square(self):
        # make sure that txt files are already in self.data_dir + 'dlpy_obj_det_test', otherwise the test will fail.
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        # non square image
        create_object_detection_table(self.s, data_path = self.data_dir + 'dlpy_obj_det_test',
                                      coord_type = 'yolo',
                                      output = 'output', image_size = (416, 512))
        # there are 11 images where all contains 3 instance.
        # If annotation files are parsed correctly, _nObjects_ column is 3 for all records.
        a = self.s.CASTable('output')
        self.assertTrue(self.s.fetch('output', fetchvars='_nObjects_').Fetch['_nObjects_'].tolist() == [3.0]*len(a))
        # check if the output size is correct
        self.assertEqual(self.s.image.summarizeimages('output').Summary.values[0][6], 416)
        self.assertEqual(self.s.image.summarizeimages('output').Summary.values[0][7], 512)

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

    def test_get_txt_annotation_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        get_txt_annotation(self.data_dir_local+'dlpy_obj_det_test', 'yolo', 416)

    def test_get_txt_annotation_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        # If there are no xml files under data_path, an error should be thrown
        self.assertRaises(DLPyError, lambda:get_txt_annotation(self.data_dir_local+'vgg', 'yolo', 416))

    def test_get_txt_annotation_non_square(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        get_txt_annotation(self.data_dir_local+'dlpy_obj_det_test', 'yolo', (416, 512))
        get_txt_annotation(self.data_dir_local + 'dlpy_obj_det_test', 'coco', (416, 512))

    def test_unify_keys(self):
        dict_1={
            'Key1':'abc',
            'key_2':'def',
            'KEY__3':'ghi',
            '_key4___':'jkl'
       }
        dict_2={
            'key1':'abc',
            'key2':'def',
            'key3':'ghi',
            'key4':'jkl'
        }
        self.assertTrue(unify_keys(dict_1)==dict_2)

    def test__ntuple(self):
        from dlpy.utils import _pair
        self.assertTrue(_pair((1,2,3))==(1,2,3))

    def test_filter_by_image_id_1(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        image_id = '1'
        self.assertRaises(ValueError, lambda:filter_by_image_id(table, image_id, filtered_name=1))


    def test_filter_by_image_id_2(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        image_id = ['1','3','4']
        filtered = filter_by_image_id(table, image_id, filtered_name=None)
        
        self.assertTrue(filtered.numrows().numrows == 3)


    def test_filter_by_image_id_3(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        image_id = 0
        self.assertRaises(ValueError,lambda:filter_by_image_id(table, image_id, filtered_name=None))


    def test_filter_by_filename_1(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        filename = 'giraffe_'
        self.assertRaises(ValueError, lambda:filter_by_filename(table, filename, filtered_name=1))
        

    def test_filter_by_filename_2(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        filename = 'giraffe_'
        filtered = filter_by_filename(table, filename, filtered_name=None)
        filtered = ImageTable.from_table(filtered)
        self.assertTrue(filtered.label_freq.loc['Giraffe'][1]>0)
        self.assertRaises(KeyError, lambda:filtered.label_freq.loc['Dolphin'])


    def test_filter_by_filename_3(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        filename = ['giraffe_', 'dolphin_']
        filtered = filter_by_filename(table, filename, filtered_name=None)
        filtered = ImageTable.from_table(filtered)
        self.assertTrue(filtered.label_freq.loc['Giraffe'][1]>0)
        self.assertTrue(filtered.label_freq.loc['Dolphin'][1]>0)
   
    def test_filter_by_filename_4(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        filename = 0
        self.assertRaises(ValueError, lambda:filter_by_filename(table, filename, filtered_name=None))

    def test_filter_by_filename_5(self): 
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir+'giraffe_dolphin_small'
        table = ImageTable.load_files(self.s, path=img_path)
        filename = [1,'text',5.3]
        self.assertRaises(ValueError, lambda:filter_by_filename(table, filename, filtered_name=None))
       
    def test_get_max_objects_1(self):
        self.assertRaises(ValueError, lambda:get_max_objects(1))

    def test_get_anchors_2(self):
        # If coord_type is 'coco', image_size must be specified.  If not, an error should be thrown
        self.assertRaises(ValueError, lambda:get_anchors(self.s,data=1,coord_type='coco', image_size=None))

    def test_parameter_2d(self):
        params = parameter_2d(param1=None, param2=None, param3=2, default_value=(5,6))
        self.assertTrue(params[0]==5)
        self.assertTrue(params[1]==2)

        params = parameter_2d(param1=None, param2=3, param3=None, default_value=(5,6))
        self.assertTrue(params[0]==3)
        self.assertTrue(params[1]==6)

    def test___init__(self):
        my_box = Box(x=2, y=3, w=4, h=5, class_type=None, confidence=1.0, 
                     image_name=None, format_type='xyxy')
        self.assertTrue(my_box.x_min==2)
        self.assertTrue(my_box.x_max==3)
        self.assertTrue(my_box.y_min==4)
        self.assertTrue(my_box.y_max==5)

    def test_plot_anchors(self):
        base_anchor_size = 16
        anchor_scale = [1.0, 2.0, 3.5]
        anchor_ratio = [4, 1, 2]
        image_size = (2000, 321)
        plot_anchors(base_anchor_size, anchor_scale, anchor_ratio, image_size)

