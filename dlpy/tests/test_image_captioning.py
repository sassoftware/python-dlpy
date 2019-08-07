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
#

import swat
import swat.utils.testing as tm
from dlpy.applications import *
from dlpy.images import ImageTable
import unittest
import os
from dlpy.image_captioning import *


class TestImageCaptioning(unittest.TestCase):

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

    # wrong dense layer
    def test_get_image_features_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s,path=img_path)
        image_table.resize(width=224)
        features_model = VGG16(self.s, width=224, height=224)
        dense_layer = 'fc10000'
        self.assertRaises(DLPyError, lambda: get_image_features(self.s,
                                                                features_model,
                                                                image_table,
                                                                dense_layer))

    # wrong target
    def test_get_image_features_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s,path=img_path)
        features_model = VGG16(self.s, width=224, height=224)
        image_table.resize(width=224)
        dense_layer = 'fc7'
        self.assertRaises(DLPyError, lambda: get_image_features(self.s,
                                                                features_model,
                                                                image_table,
                                                                dense_layer,
                                                                target='not_column'))

    # works
    def test_get_image_features_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s,path=img_path)
        image_table.resize(width=224)
        features_model = VGG16(self.s,
                               width=224,
                               height=224,
                               pre_trained_weights=True,
                               pre_trained_weights_file=self.data_dir+'VGG_ILSVRC_16_layers.caffemodel.h5')
        dense_layer = 'fc7'
        self.assertTrue(get_image_features(self.s,features_model,image_table,dense_layer) is not None)

    # works
    def test_captions_table_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        self.assertTrue(create_captions_table(self.s, captions_file) is not None)

    # captions file doesn't exist
    def test_captions_table_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'no_file.txt'
        self.assertRaises(FileNotFoundError, lambda:create_captions_table(self.s, captions_file))

    # incorrect delimiter
    def test_captions_table_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAK is not set in the environment variables")
        delimiter = ','
        captions_file = self.data_dir_local + 'image_captions.txt'
        self.assertRaises(DLPyError, lambda:create_captions_table(self.s, captions_file,delimiter=delimiter))

    # works
    def test_object_table_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s, path=img_path)
        image_table.resize(width=416)

        word_embeddings = self.data_dir_local + 'word_embeddings.txt'
        detection_model = Model(self.s)
        detection_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        detection_model.load_weights(self.data_dir + 'YoloV2_Multisize_weights.sashdat')

        tbl = create_embeddings_from_object_detection(self.s,
                                                      image_table,
                                                      detection_model,
                                                      word_embeddings)
        self.assertTrue(tbl is not None)

    # word embeddings file doesn't exist
    def test_object_table_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s, path=img_path)
        image_table.resize(width=416)

        word_embeddings = self.data_dir + 'no_file.txt'
        detection_model = Model(self.s)
        detection_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        detection_model.load_weights(self.data_dir + 'YoloV2_Multisize_weights.sashdat')

        self.assertRaises(DLPyError, lambda:create_embeddings_from_object_detection(self.s,
                                                                                    image_table,
                                                                                    detection_model,
                                                                                    word_embeddings))

    def test_object_table_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        image_table = create_captions_table(self.s, captions_file)

        word_embeddings = self.data_dir + 'word_embeddings.txt'
        detection_model = Model(self.s)
        detection_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        detection_model.load_weights(self.data_dir + 'YoloV2_Multisize_weights.sashdat')

        self.assertRaises(DLPyError, lambda:create_embeddings_from_object_detection(self.s,
                                                                                    image_table,
                                                                                    detection_model,
                                                                                    word_embeddings))

    #works
    def test_reshape_columns_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        table = create_captions_table(self.s, captions_file)
        self.assertTrue(reshape_caption_columns(self.s, table) is not None)

    # wrong caption_col_name
    def test_reshape_columns_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        table = create_captions_table(self.s, captions_file)

        self.assertRaises(DLPyError, lambda:reshape_caption_columns(self.s, table, caption_col_name='random'))

    # wrong number of captions
    def test_reshape_columns_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        table = create_captions_table(self.s, captions_file)
        num_caps = 10
        self.assertRaises(DLPyError, lambda:reshape_caption_columns(self.s, table, num_captions=num_caps))

    # # works w/o object detection
    def test_captioning_table_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s,path=img_path)
        image_table.resize(width=224)
        captions_file = self.data_dir_local + 'image_captions.txt'
        features_model = VGG16(self.s,
                               width=224,
                               height=224,
                               pre_trained_weights=True,
                               pre_trained_weights_file=self.data_dir+'VGG_ILSVRC_16_layers.caffemodel.h5')

        self.assertTrue(create_captioning_table(self.s, image_table, features_model, captions_file) is not None)

    # # object detection w/o word_embeddings (error)
    def test_captioning_table_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s, path=img_path)
        image_table.resize(width=224)
        captions_file = self.data_dir_local + 'image_captions.txt'
        features_model = VGG16(self.s,
                               width=224,
                               height=224,
                               pre_trained_weights=True,
                               pre_trained_weights_file=self.data_dir+'VGG_ILSVRC_16_layers.caffemodel.h5')
        detection_model = Model(self.s)
        detection_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        detection_model.load_weights(self.data_dir + 'YoloV2_Multisize_weights.sashdat')
        self.assertRaises(DLPyError,lambda:
                          create_captioning_table(self.s,
                                                  image_table,
                                                  features_model,
                                                  captions_file,
                                                  obj_detect_model=detection_model))

    # object detection w/ word_embeddings
    def test_captioning_table_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        img_path = self.data_dir + 'imageCaptioning_images'
        image_table = ImageTable.load_files(self.s,path=img_path)
        image_table.resize(width=224)
        captions_file = self.data_dir_local + 'image_captions.txt'
        features_model = VGG16(self.s,
                               width=224,
                               height=224,
                               pre_trained_weights=True,
                               pre_trained_weights_file=self.data_dir+'VGG_ILSVRC_16_layers.caffemodel.h5')

        detection_model = Model(self.s)
        detection_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        detection_model.load_weights(self.data_dir + 'YoloV2_Multisize_weights.sashdat')
        word_embeddings = self.data_dir_local + 'word_embeddings.txt'
        self.assertTrue(create_captioning_table(self.s,
                                                image_table,
                                                features_model,
                                                captions_file,
                                                obj_detect_model=detection_model,
                                                word_embeddings_file=word_embeddings) is not None)

    # works
    def test_ImageCaptioning_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        model1 = ImageCaptioning(self.s,rnn_type='GRU')
        self.assertTrue(model1 is not None)

        model2 = ImageCaptioning(self.s,rnn_type='RNN')
        self.assertTrue(model2 is not None)

        model3 = ImageCaptioning(self.s)
        self.assertTrue(model3 is not None)

    def test_ImageCaptioning_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self,"DLPY_DATA_DIR is not set in the environment variables")

        model1 = ImageCaptioning(self.s,num_blocks=1)
        self.assertTrue(model1 is not None)

        model2 = ImageCaptioning(self.s,num_blocks=10)
        self.assertTrue(model2 is not None)

        model3 = ImageCaptioning(self.s,num_blocks=100)
        self.assertTrue(model3 is not None)

    def test_ImageCaptioning_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self,"DLPY_DATA_DIR is not set in the environment variables")

        self.assertRaises(DLPyError,lambda: ImageCaptioning(self.s,num_blocks=0))

    def test_results_to_dict_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        new_dict = dict(_DL_Pred_=['predict1','predict2','predict3'],
                        caption=['capt1','capt2','capt3'],
                        _filename_0=['file1','file2','file3'])
        results = CASTable.from_dict(self.s,new_dict)

        self.assertTrue(scored_results_to_dict(results) is not None)

    def test_results_to_dict_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        result_tbl = self.s.CASTable('not_a_table')
        self.assertRaises(DLPyError,lambda:
                                    scored_results_to_dict(result_tbl))

    def test_capt_len_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        self.assertEqual(get_max_capt_len(captions_file, delimiter='\t'),21)

    #     wrong delimiter
    def test_capt_len_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")
        captions_file = self.data_dir_local + 'image_captions.txt'
        self.assertRaises(DLPyError, lambda:
                          get_max_capt_len(captions_file, delimiter=','))

    # bad file
    def test_capt_len_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        captions_file = "doesnt/exist.txt"
        self.assertRaises(FileNotFoundError,lambda:
                          get_max_capt_len(captions_file,delimiter='\t'))