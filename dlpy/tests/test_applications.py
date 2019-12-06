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
from dlpy.utils import file_exist_on_server
import unittest
import json
import os
import pandas as pd


class TestApplications(unittest.TestCase):
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

        filename = os.path.join('datasources', 'sample_syntax_for_test.json')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)
        with open(full_filename) as f:
            cls.sample_syntax = json.load(f)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_resnet50_caffe(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'ResNet-50-model.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = ResNet50_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                               pre_trained_weights_file=self.data_dir + 'ResNet-50-model.caffemodel.h5',
                               pre_trained_weights=True,
                               include_top=False,
                               n_classes=120,
                               random_crop='unique')
        model.print_summary()

        model = ResNet50_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                               pre_trained_weights_file=self.data_dir + 'ResNet-50-model.caffemodel.h5',
                               pre_trained_weights=True,
                               include_top=False,
                               n_classes=120,
                               random_crop=None,
                               offsets=None)
        model.print_summary()

        # test random_mutation and crop on VDMML 8.4
        model = ResNet50_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                               pre_trained_weights_file=self.data_dir + 'ResNet-50-model.caffemodel.h5',
                               pre_trained_weights=True,
                               include_top=False,
                               n_classes=120,
                               random_crop='RESIZETHENCROP',
                               random_mutation='random',
                               offsets=None)

    def test_resnet50_caffe_caslib_msg(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'ResNet-50-model.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = ResNet50_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                               pre_trained_weights_file=self.data_dir + 'ResNet-50-model.caffemodel.h5',
                               pre_trained_weights=True,
                               include_top=False,
                               n_classes=120,
                               random_crop='unique')

        model.print_summary()

    def test_resnet50_layerid(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        model = ResNet50_Caffe(self.s)
        model.print_summary()
        model.print_summary()

    def test_resnet101_caffe(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'ResNet-101-model.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = ResNet101_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-101-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop='unique')
        model.print_summary()

        model = ResNet101_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-101-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop=None,
                                offsets=None)
        model.print_summary()

        self.assertRaises(ValueError,
                          lambda: ResNet101_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                                  pre_trained_weights_file=self.data_dir + 'ResNet-101-model.caffemodel.h5',
                                                  pre_trained_weights=True,
                                                  include_top=False,
                                                  n_classes=120,
                                                  random_crop='wrong_val'))

        # test random_mutation and crop on VDMML 8.4
        model = ResNet101_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-101-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop='RESIZETHENCROP',
                                random_mutation='random',
                                offsets=None)
        model.print_summary()

    def test_resnet152_caffe(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'ResNet-152-model.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = ResNet152_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-152-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop='unique')
        model.print_summary()

        model = ResNet152_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-152-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop=None,
                                offsets=None)
        model.print_summary()

        self.assertRaises(ValueError,
                          lambda: ResNet152_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                                  pre_trained_weights_file=self.data_dir + 'ResNet-152-model.caffemodel.h5',
                                                  pre_trained_weights=True,
                                                  include_top=False,
                                                  n_classes=120,
                                                  random_crop='wrong_val'))
        # test random_mutation and crop on VDMML 8.4
        model = ResNet152_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                                pre_trained_weights_file=self.data_dir + 'ResNet-152-model.caffemodel.h5',
                                pre_trained_weights=True,
                                include_top=False,
                                n_classes=120,
                                random_crop='RESIZETHENCROP',
                                random_mutation='random',
                                offsets=None)
        model.print_summary()

    def test_lenet5(self):
        from dlpy.applications import LeNet5
        model = LeNet5(self.s)
        model.print_summary()

    def test_vgg11(self):
        from dlpy.applications import VGG11
        model = VGG11(self.s)
        model.print_summary()

        # test random_crop and mutation
        model1 = VGG11(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random')

        model1.print_summary()
        res1 = self.s.fetch(table=model1.model_name, sortby='_dllayerid_')
        print(res1)

        self.assertEqual(res1['Fetch'].iloc[10, 3], 4)
        self.assertEqual(res1['Fetch'].iloc[4, 3], 2)

    def test_vgg13(self):
        from dlpy.applications import VGG13
        model = VGG13(self.s)
        model.print_summary()

        # test random_crop and mutation
        model1 = VGG13(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random')

        model1.print_summary()
        res1 = self.s.fetch(table=model1.model_name, sortby='_dllayerid_')
        print(res1)

        self.assertEqual(res1['Fetch'].iloc[10, 3], 4)
        self.assertEqual(res1['Fetch'].iloc[4, 3], 2)

    def test_vgg16(self):
        from dlpy.applications import VGG16

        pd.set_option('display.max_columns', 40)

        model = VGG16(self.s)
        model.print_summary()
        res = self.s.fetch(table=model.model_name, sortby='_dllayerid_')
        print(res)

        # test random_crop and mutation
        model1 = VGG16(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random',
                       include_top=True)
        model1.print_summary()
        res1 = self.s.fetch(table=model1.model_name, sortby='_dllayerid_')
        print(res1)

        self.assertEqual(res1['Fetch'].iloc[10, 3], 4)
        self.assertEqual(res1['Fetch'].iloc[4, 3], 2)

    def test_vgg16_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model1 = VGG16(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=(103.939, 116.779, 123.68),
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5',
                       include_top=True)
        model1.print_summary()

        model2 = VGG16(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop=None,
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5',
                       include_top=True)
        model2.print_summary()

        self.assertRaises(ValueError,
                          lambda: VGG16(self.s, model_table='VGG16',
                                        n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                                        offsets=None,
                                        random_crop='wrong_val',
                                        pre_trained_weights=True,
                                        pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5',
                                        include_top=True))

        # test random_crop and mutation
        model3 = VGG16(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random',
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5',
                       include_top=True)
        model3.print_summary()

    def test_vgg19(self):
        from dlpy.applications import VGG19
        model = VGG19(self.s)
        model.print_summary()

        # test random_crop and mutation
        model1 = VGG19(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random',
                       include_top=True)
        model1.print_summary()
        res1 = self.s.fetch(table=model1.model_name, sortby='_dllayerid_')
        print(res1)

        self.assertEqual(res1['Fetch'].iloc[10, 3], 4)
        self.assertEqual(res1['Fetch'].iloc[4, 3], 2)

    def test_vgg19_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'VGG_ILSVRC_19_layers.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model1 = VGG19(self.s, model_table='VGG19', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=(103.939, 116.779, 123.68),
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_19_layers.caffemodel.h5',
                       include_top=True)

        model2 = VGG19(self.s, model_table='VGG19', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop=None,
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_19_layers.caffemodel.h5',
                       include_top=True)

        self.assertRaises(ValueError,
                          lambda: VGG19(self.s, model_table='VGG19',
                                        n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                                        offsets=None,
                                        random_crop='wrong_val',
                                        pre_trained_weights=True,
                                        pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_19_layers.caffemodel.h5',
                                        include_top=True))
        # test random_crop and mutation
        model3 = VGG19(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                       width=224, height=224, scale=1,
                       offsets=None,
                       random_crop='unique',
                       random_flip='hv',
                       random_mutation='random',
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir + 'VGG_ILSVRC_16_layers.caffemodel.h5',
                       include_top=True)
        model3.print_summary()

    def test_resnet18(self):
        from dlpy.applications import ResNet18_SAS
        model = ResNet18_SAS(self.s)
        model.print_summary()

    def test_resnet18_2(self):
        from dlpy.applications import ResNet18_Caffe
        model = ResNet18_Caffe(self.s)
        model.print_summary()

    def test_resnet34(self):
        from dlpy.applications import ResNet34_SAS
        model = ResNet34_SAS(self.s)
        model.print_summary()

    def test_resnet34_2(self):
        from dlpy.applications import ResNet34_Caffe
        model = ResNet34_Caffe(self.s)
        model.print_summary()

    def test_resnet50(self):
        from dlpy.applications import ResNet50_SAS
        model = ResNet50_SAS(self.s)
        model.print_summary()

    def test_resnet50_2(self):
        from dlpy.applications import ResNet50_Caffe
        model = ResNet50_Caffe(self.s)
        model.print_summary()

    # test resnet50 with reshape
    def test_resnet50_3(self):
        from dlpy.applications import ResNet50_Caffe

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'ResNet-50-model.caffemodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        reshape = Reshape(width=224, height=224, depth=3, order='WHD')
        model = ResNet50_Caffe(self.s, reshape_after_input=reshape)
        model.print_summary()

        # test it with pretrained weights
        model1 = ResNet50_Caffe(self.s, model_table='Resnet50', n_classes=1000, n_channels=3,
                                width=224, height=224, scale=1,
                                offsets=None,
                                random_crop='unique',
                                random_flip='hv',
                                random_mutation='random',
                                pre_trained_weights=True,
                                pre_trained_weights_file=self.data_dir + 'ResNet-50-model.caffemodel.h5',
                                include_top=True,
                                reshape_after_input=reshape)
        res = model1.print_summary()
        print(res)
        self.assertEqual(res.iloc[1, 6][0], 224)
        self.assertEqual(res.iloc[1, 6][1], 224)
        self.assertEqual(res.iloc[1, 6][2], 3)

    # test resnet50 with the wrong reshape layer
    def test_resnet50_4(self):
        from dlpy.applications import ResNet50_Caffe

        reshape = Pooling(width=2, height=2, stride=2)
        self.assertRaises(DLPyError, lambda: ResNet50_Caffe(self.s, reshape_after_input=reshape))

    def test_resnet101(self):
        from dlpy.applications import ResNet101_SAS
        model = ResNet101_SAS(self.s)
        model.print_summary()

    def test_resnet101_2(self):
        from dlpy.applications import ResNet101_Caffe
        model = ResNet101_Caffe(self.s)
        model.print_summary()

    def test_resnet152(self):
        from dlpy.applications import ResNet152_SAS
        model = ResNet152_SAS(self.s)
        model.print_summary()

    def test_resnet152_2(self):
        from dlpy.applications import ResNet152_Caffe
        model = ResNet152_Caffe(self.s)
        model.print_summary()

    def test_resnet_wide(self):
        from dlpy.applications import ResNet_Wide
        model = ResNet_Wide(self.s, number_of_blocks=1)
        print(model.summary.iloc[:, 5])

    def test_densenet(self):
        from dlpy.applications import DenseNet
        model = DenseNet(self.s)
        model.print_summary()

    def test_densenet_2(self):
        from dlpy.applications import DenseNet
        model = DenseNet(self.s, conv_channel=32)
        model.print_summary()

    def test_densenet_3(self):
        from dlpy.applications import DenseNet
        model = DenseNet(self.s, n_blocks=7)
        model.print_summary()

    def test_densenet_4(self):
        from dlpy.applications import DenseNet
        model = DenseNet(self.s, n_blocks=2)
        model.print_summary()

    def test_densenet121(self):
        from dlpy.applications import DenseNet121
        model = DenseNet121(self.s)
        model.print_summary()

    def test_densenet121_1(self):
        from dlpy.applications import DenseNet121
        model = DenseNet121(self.s, n_cells=[1, 1, 1, 1])
        model.print_summary()

    def test_densenet121_2(self):
        from dlpy.applications import DenseNet121
        model = DenseNet121(self.s, conv_channel=1)
        model.print_summary()

    def test_darknet_ref(self):
        from dlpy.applications import Darknet_Reference
        model = Darknet_Reference(self.s)
        model.print_summary()

    def test_darknet(self):
        from dlpy.applications import Darknet
        model = Darknet(self.s)
        model.print_summary()

    def test_yolov1(self):
        from dlpy.applications import YoloV1
        model = YoloV1(self.s)
        model.print_summary()

    def test_yolov2(self):
        with self.assertRaises(DLPyError):
            from dlpy.applications import YoloV2
            anchors = []
            anchors.append(1)
            anchors.append(1)
            model = YoloV2(self.s, anchors)
            model.print_summary()

    def test_yolov2_2(self):
        from dlpy.applications import YoloV2
        anchors = []
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=1)
        model.print_summary()

    def test_yolov2_3(self):
        from dlpy.applications import YoloV2
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=4)
        model.print_summary()

    def test_yolov2_4(self):
        from dlpy.applications import YoloV2
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=1)
        model.print_summary()

    def test_yolov2_multi(self):
        from dlpy.applications import YoloV2_MultiSize
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2_MultiSize(self.s, anchors, predictions_per_grid=2)
        model.print_summary()

    def test_yolov2_multi_2(self):
        from dlpy.applications import YoloV2_MultiSize
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2_MultiSize(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=1)
        model.print_summary()

    def test_yolov2_tiny(self):
        from dlpy.applications import Tiny_YoloV2
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = Tiny_YoloV2(self.s, anchors, predictions_per_grid=2)
        model.print_summary()

    def test_yolov2_tiny_2(self):
        from dlpy.applications import Tiny_YoloV2
        anchors = []
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = Tiny_YoloV2(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=1)
        model.print_summary()

    def test_yolov1_tiny(self):
        from dlpy.applications import Tiny_YoloV1
        model = Tiny_YoloV1(self.s)
        model.print_summary()

    def test_yolov1_tiny_2(self):
        from dlpy.applications import Tiny_YoloV1
        model = Tiny_YoloV1(self.s, n_classes=3)
        model.print_summary()

    def test_yolov1_tiny(self):
        from dlpy.applications import Tiny_YoloV1
        model = Tiny_YoloV1(self.s)
        model.print_summary()

    def test_yolov2_tiny_2(self):
        from dlpy.applications import Tiny_YoloV1
        model = Tiny_YoloV1(self.s, n_classes=7)
        model.print_summary()

    def test_darknet(self):
        from dlpy.applications import Darknet
        model = Darknet(self.s)
        model.print_summary()

    def test_darknet_1(self):
        from dlpy.applications import Darknet
        model = Darknet(self.s, n_classes=4)
        model.print_summary()

    def test_inceptionv3(self):
        from dlpy.applications import InceptionV3
        model = InceptionV3(self.s)
        model.print_summary()

    def test_inceptionv3_1(self):
        from dlpy.applications import InceptionV3
        model = InceptionV3(self.s, n_classes=3)
        model.print_summary()

    def test_inceptionv3_2(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'InceptionV3_weights.kerasmodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        from dlpy.applications import InceptionV3
        model = InceptionV3(self.s,
                            model_table='INCEPTIONV3',
                            n_classes=1000,
                            n_channels=3,
                            width=299,
                            height=299,
                            scale=1,
                            offsets=(103.939, 116.779, 123.68),
                            pre_trained_weights=True,
                            pre_trained_weights_file=self.data_dir + 'InceptionV3_weights.kerasmodel.h5',
                            include_top=True)
        model.print_summary()

        # test random_crop and mutation
        model2 = InceptionV3(self.s, model_table='VGG16', n_classes=1000, n_channels=3,
                             width=299, height=299, scale=1,
                             offsets=None,
                             random_crop='unique',
                             random_flip='hv',
                             random_mutation='random',
                             pre_trained_weights=True,
                             pre_trained_weights_file=self.data_dir + 'InceptionV3_weights.kerasmodel.h5',
                             include_top=True)
        model2.print_summary()

    def test_inceptionv3_3(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'InceptionV3_weights.kerasmodel.h5'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        from dlpy.applications import InceptionV3
        model = InceptionV3(self.s,
                            model_table='INCEPTIONV3',
                            n_classes=3,
                            n_channels=3,
                            width=299,
                            height=299,
                            scale=1,
                            offsets=(103.939, 116.779, 123.68),
                            pre_trained_weights=True,
                            pre_trained_weights_file=self.data_dir + 'InceptionV3_weights.kerasmodel.h5',
                            include_top=False)
        model.print_summary()

    def test_text_classification(self):
        from dlpy.applications import TextClassification
        model = TextClassification(self.s)
        model.print_summary()

    def test_text_classification_1(self):
        from dlpy.applications import TextClassification
        model = TextClassification(self.s, n_blocks=1)
        model.print_summary()

    def test_text_generation(self):
        from dlpy.applications import TextGeneration
        model = TextGeneration(self.s)
        model.print_summary()

    def test_text_generation_1(self):
        with self.assertRaises(DLPyError):
            from dlpy.applications import TextGeneration
            model = TextGeneration(self.s, n_blocks=1)
            model.print_summary()

    def test_sequence_labeling(self):
        from dlpy.applications import SequenceLabeling
        model = SequenceLabeling(self.s)
        model.print_summary()

    def test_sequence_labeling_1(self):
        from dlpy.applications import SequenceLabeling
        model = SequenceLabeling(self.s, n_blocks=1)
        model.print_summary()

    def test_speech_recognition(self):
        from dlpy.applications import SpeechRecognition
        model = SpeechRecognition(self.s)
        model.print_summary()

    def test_speech_recognition_1(self):
        from dlpy.applications import SpeechRecognition
        model = SpeechRecognition(self.s, n_blocks=1)
        model.print_summary()

    def test_fast_rcnn(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s)
        model.print_summary()

    def test_fast_rcnn_2(self):
        from dlpy.applications import Faster_RCNN
        anchor_num_to_sample = 64
        anchor_ratio = [2312312, 2, 2]
        anchor_scale = [1.2, 2.3, 3.4, 5.6]
        coord_type = 'rect'
        model = Faster_RCNN(self.s, model_table='fast', anchor_num_to_sample=anchor_num_to_sample,
                            anchor_ratio=anchor_ratio, anchor_scale=anchor_scale, coord_type=coord_type)
        self.assertTrue(model.layers[20].config == self.sample_syntax['faster_rcnn1'])
        model.print_summary()

    def test_fast_rcnn_3(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet50')
        model.print_summary()

    def test_fast_rcnn_4(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet50', number_of_neurons_in_fc=1024)
        model.print_summary()

    def test_fast_rcnn_5(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet50', number_of_neurons_in_fc=[1024, 1024])
        model.print_summary()

    def test_fast_rcnn_6(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet50', number_of_neurons_in_fc=[1024])
        model.print_summary()

    def test_fast_rcnn_7(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet18', number_of_neurons_in_fc=[1024, 1024])
        model.print_summary()

    def test_fast_rcnn_8(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='resnet34')
        model.print_summary()

    def test_fast_rcnn_9(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='mobilenetv1')
        model.print_summary()

    def test_fast_rcnn_10(self):
        from dlpy.applications import Faster_RCNN
        model = Faster_RCNN(self.s, backbone='mobilenetv2', width=123, height=321)
        model.print_summary()

    def test_mobilenetv1(self):
        from dlpy.applications import MobileNetV1
        model = MobileNetV1(self.s, n_classes=2, n_channels=3, depth_multiplier=10, alpha=2)
        self.assertTrue(len(model.layers) == 57)
        self.assertTrue(model.layers[49]._output_size == (7, 7, 2048))
        model.print_summary()
        self.assertEqual(model.summary.iloc[7, -1], 36126720)

    def test_mobilenetv2(self):
        from dlpy.applications import MobileNetV2
        model = MobileNetV2(self.s, n_classes=2, n_channels=3, alpha=2)
        self.assertTrue(len(model.layers) == 117)
        self.assertTrue(model.layers[112]._output_size == (7, 7, 640))
        model.print_summary()

    def test_shufflenetv1(self):
        from dlpy.applications import ShuffleNetV1
        model = ShuffleNetV1(self.s, n_classes=2, n_channels=3, scale_factor=2, num_shuffle_units=[2, 2, 3, 4],
                             bottleneck_ratio=0.4, groups=2)
        self.assertTrue(len(model.layers) == 130)
        self.assertTrue(model.layers[127]._output_size == (4, 4, 3200))
        model.print_summary()

    def test_unet(self):
        from dlpy.applications import UNet
        model = UNet(self.s, width=1024, height=1024, offsets=[1.25], scale=0.0002)
        self.assertTrue(len(model.layers) == 33)
        self.assertTrue(model.layers[12].output_size == (64, 64, 512))
        model.print_summary()
        # transpose conv print summary numerical check
        model = UNet(self.s, width=256, height=256, offsets=[1.25], scale=0.0002)
        model.print_summary()
        self.assertEqual(model.summary.iloc[23, -1], 4831838208)

    def test_unet_with_bn(self):
        from dlpy.applications import UNet
        # append bn layer right after conv
        model = UNet(self.s, width=1024, height=1024, offsets=[1.25], scale=0.0002,
                     bn_after_convolutions=True)
        self.assertTrue(len(model.layers) == 33 + 9 * 2)
        self.assertTrue(model.layers[12].output_size == (256, 256, 256))
        model.print_summary()

    def test_mobilenetv2_onnx_1(self):
        from dlpy.applications import MobileNetV2_ONNX

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'MobileNetV2_ONNX.sashdat'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = MobileNetV2_ONNX(self.s,
                                 model_file=file_dependency,
                                 width=224, height=224,
                                 n_classes=10, include_top=False)
        model.print_summary()

    def test_mobilenetv2_onnx_2(self):
        from dlpy.applications import MobileNetV2_ONNX

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'MobileNetV2_ONNX.sashdat'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = MobileNetV2_ONNX(self.s,
                                 model_file=file_dependency,
                                 width=224, height=224,
                                 n_classes=1000, include_top=True)
        model.print_summary()

    def test_densenet121_onnx_1(self):
        from dlpy.applications import DenseNet121_ONNX

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'DenseNet121_ONNX.sashdat'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = DenseNet121_ONNX(self.s,
                                 model_file=file_dependency,
                                 width=224, height=224,
                                 n_classes=10, include_top=False)
        model.print_summary()

    def test_densenet121_onnx_2(self):
        from dlpy.applications import DenseNet121_ONNX

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        file_dependency = self.data_dir + 'DenseNet121_ONNX.sashdat'
        if not file_exist_on_server(self.s, file_dependency):
            unittest.TestCase.skipTest(self, "File, {}, not found.".format(file_dependency))

        model = DenseNet121_ONNX(self.s,
                                 model_file=file_dependency,
                                 width=224, height=224,
                                 n_classes=1000, include_top=True)
        model.print_summary()
