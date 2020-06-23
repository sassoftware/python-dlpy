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
import csv
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
    code_cov_skip = 0

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.s)
        self.server_sep = '\\'
        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            self.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if self.data_dir_local.startswith('/'):
                sep_ = '/'
            else:
                sep_ = '\\'
            if self.data_dir_local.endswith(sep_):
                self.data_dir_local = self.data_dir_local[:-1]
            self.data_dir_local += sep_

        if 'CODE_COV_SKIP' in os.environ:
            self.code_cov_skip = 1


    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
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
                                                                           data_path=self.data_dir+'dlpy_obj_det_test',
                                                                           coord_type='invalid_val',
                                                                           output='output'))

    def test_create_object_detection_table_4(self):
        # make sure that txt files are already in self.data_dir + 'dlpy_obj_det_test', otherwise the test will fail.
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        create_object_detection_table(self.s, data_path = self.data_dir + 'dlpy_obj_det_test',
                                      coord_type='yolo',
                                      output='output')

        a = self.s.CASTable('output')
        from dlpy.utils import get_info_for_object_detection
        c, m = get_info_for_object_detection( self.s, a)
        d = {'Black King': 10.0,
             'White King': 9.0,
             'White Rook': 10.0,
             'White Kinghuh': 1.0}

        self.assertEqual(len(d), 4)
        self.assertEqual(d, c)
        self.assertEqual(m, 3)

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

    def test_get_txt_annotation_name_file(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        get_txt_annotation(self.data_dir_local+'dlpy_obj_det_test', 'yolo', (416, 512),
                           name_file = os.path.join(self.data_dir_local, 'dlpy_obj_det_test', 'coco.names'))

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

    def test_box_xyxy(self):
        my_box = Box(x=2, y=3, w=4, h=5, class_type=None, confidence=1.0, 
                     image_name=None, format_type='xyxy')
        self.assertTrue(my_box.x_min==2)
        self.assertTrue(my_box.y_min==3)
        self.assertTrue(my_box.x_max==4)
        self.assertTrue(my_box.y_max==5)

    def test_plot_anchors(self):
        base_anchor_size = 16
        anchor_scale = [1.0, 2.0, 3.5]
        anchor_ratio = [4, 1, 2]
        image_size = (2000, 321)
        plot_anchors(base_anchor_size, anchor_scale, anchor_ratio, image_size)

    def test_create_metadata_table_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if self.code_cov_skip == 1:
            unittest.TestCase.skipTest(self, "Test is skipped in code coverage analysis")

        create_metadata_table(self.s, folder=self.data_dir)

    def test_create_metadata_table_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if self.code_cov_skip == 1:
            unittest.TestCase.skipTest(self, "Test is skipped in code coverage analysis")
        create_metadata_table(self.s, folder=self.data_dir, extensions_to_filter=['.jpg'])

    def test_create_metadata_table_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if self.code_cov_skip == 1:
            unittest.TestCase.skipTest(self, "Test is skipped in code coverage analysis")

        create_metadata_table(self.s, folder='/random/location')

        with self.assertRaises(DLPyError):
            create_metadata_table(self.s, folder='dlpy', caslib='random_caslib')

    def test_create_segmentation_table(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        server_type = get_cas_host_type(self.s).lower()

        if server_type.startswith("lin") or server_type.startswith("osx"):
            sep = '/'
        else:
            sep = '\\'

        tbl = create_segmentation_table(self.s,
                                        path_to_images=self.data_dir+'segmentation_data'+sep+'raw',
                                        path_to_ground_truth=self.data_dir+'segmentation_data'+sep+'mask')

    def test_caslibify_context(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        server_type = get_cas_host_type(self.s).lower()

        if server_type.startswith("lin") or server_type.startswith("osx"):
            sep = '/'
        else:
            sep = '\\'
        tmp_caslib = None
        # save task
        with caslibify_context(self.s, self.data_dir+'segmentation_data'+sep+'raw', 'load') as (caslib, path):
            tmp_caslib = caslib
            df = self.s.caslibinfo().CASLibInfo['Name']
            # in the context, the new caslib should be created
            self.assertEqual(df[df == caslib].shape[0], 1)
            try:
                raise DLPyError('force to throw an error')
            except DLPyError:
                pass
        # expect the caslib is removed even if the error occurs
        df = self.s.caslibinfo().CASLibInfo['Name']
        self.assertEqual(df[df == tmp_caslib].shape[0], 0)

        # save task
        with caslibify_context(self.s, self.data_dir+'segmentation_data'+sep+'raw', 'save') as (caslib, path):
            tmp_caslib = caslib
            df = self.s.caslibinfo().CASLibInfo['Name']
            # in the context, the new caslib should be created
            self.assertEqual(df[df == caslib].shape[0], 1)
            try:
                raise DLPyError('force to throw an error')
            except DLPyError:
                pass
        # expect the caslib is removed even if the error occurs
        df = self.s.caslibinfo().CASLibInfo['Name']
        self.assertEqual(df[df == tmp_caslib].shape[0], 0)

    def test_caslibify_subdirectory_permission(self):
        self.s.addcaslib(path = self.data_dir, name='data', subdirectories=False)
        #self.assertRaises(DLPyError, lambda: caslibify(self.s, path = self.data_dir + 'segmentation_data'))
        self.s.dropcaslib(caslib='data')

    def test_caslibify_context_subdirectory_permission(self):
        self.s.addcaslib(path = self.data_dir, name='data', subdirectories=False)
        try:
            with caslibify_context(self.s, path = self.data_dir + 'segmentation_data'):
                a = 1+1
        except DLPyError:
            self.s.dropcaslib(caslib = 'data')
            return
        self.s.dropcaslib(caslib = 'data')
        #raise DLPyError('caslibify_context() expected to throw a DLPyError')

    def test_user_defined_labels(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        server_type = get_cas_host_type(self.s).lower()

        if server_type.startswith("lin") or server_type.startswith("osx"):
            sep = '/'
        else:
            sep = '\\'
            
        # write user-defined label table to CSV file
        levels = [
                  '&', # : 0,
                  'A', # : 1,
                  'B', # : 2,
                  'C', # : 3,
                  'D', # : 4,
                  'E', # : 5,
                  'F', # : 6,
                  'G', # : 7,
                  'H', # : 8,
                  'I', # : 9,
                  'J', # : 10,
                  'K', # : 11,
                  'L', # : 12,
                  'M', # : 13,
                  'N', # : 14,
                  'O', # : 15,
                  'P', # : 16,
                  'Q', # : 17,
                  'R', # : 18,
                  'S', # : 19,
                  'T', # : 20,
                  'U', # : 21,
                  'V', # : 22,
                  'W', # : 23,
                  'X', # : 24,
                  'Y', # : 25,
                  'Z', # : 26,
                  '\'', # : 27
                  ' ', # : 28
                ]

        # save labels file to local data directory
        header = ['label_id'] + ['label']

        import os
        label_file_name = os.path.join(self.data_dir_local, 'rnn_import_labels.csv')
        with open(label_file_name, 'w+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)

            for ii, lval in enumerate(levels):
                row = [str(ii)] + [lval]
                writer.writerow(row)

        # test using maximum label length in CSV to set uploaded table label length
        label_table1 = get_user_defined_labels_table(self.s, label_file_name, None)

        self.assertTrue(label_table1 is not None)

        # test specifying label length to set uploaded table label length
        label_table2 = get_user_defined_labels_table(self.s, label_file_name, 6)

        self.assertTrue(label_table2 is not None)

        os.remove(label_file_name)

    def test_print_predefined_models(self):
        print_predefined_models()

    def test_check_layer_type(self):
        from dlpy.layers import Conv2DTranspose, Conv2d
        layer = Conv2DTranspose(10)
        self.assertRaises(DLPyError, lambda: check_layer_class(layer, Conv2d))

    def test_create_instance_segmentation_castable(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        try:
            import cv2
            create_instance_segmentation_table(self.s, coord_type = 'yolo', output = 'instance_seg',
                                           data_path = self.data_dir + 'instance_segmentation_data',
                                           local_path = os.path.join(self.data_dir_local, 'instance_segmentation_data'))
            self.assertTrue(self.s.numrows('instance_seg').numrows == 1)
        except:
            unittest.TestCase.skipTest(self, "no cv2")

    def test_file_exist_on_server(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        check_file = self.data_dir + 'vgg16.sashdat'
        self.assertTrue(file_exist_on_server(self.s, file=check_file))


if __name__ == '__main__':
    unittest.main()
