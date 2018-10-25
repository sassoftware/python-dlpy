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
import unittest


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

        model = ResNet50_Caffe(self.s, n_channels=3, height=224, random_flip='HV',
                               pre_trained_weights_file=self.data_dir+'ResNet-50-model.caffemodel.h5',
                               pre_trained_weights=True,
                               include_top=False,
                               n_classes=120,
                               random_crop='unique')
        model.print_summary()

    def test_lenet5(self):
        from dlpy.applications import LeNet5
        model = LeNet5(self.s)
        model.print_summary()

    def test_vgg11(self):
        from dlpy.applications import VGG11
        model = VGG11(self.s)
        model.print_summary()

    def test_vgg13(self):
        from dlpy.applications import VGG13
        model = VGG13(self.s)
        model.print_summary()

    def test_vgg16(self):
        from dlpy.applications import VGG16
        model = VGG16(self.s)
        model.print_summary()

    def test_vgg16_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        model1 = VGG16(self.s, model_table='VGG16', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
                       offsets=(103.939, 116.779, 123.68),
                       pre_trained_weights=True,
                       pre_trained_weights_file=self.data_dir+'VGG_ILSVRC_16_layers.caffemodel.h5',
                       include_top=True)

    def test_vgg19(self):
        from dlpy.applications import VGG19
        model = VGG19(self.s)
        model.print_summary()

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
            anchors=[]
            anchors.append(1)
            anchors.append(1)
            model = YoloV2(self.s, anchors)
            model.print_summary()

    def test_yolov2_2(self):
        from dlpy.applications import YoloV2
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=1)
        model.print_summary()

    def test_yolov2_3(self):
        from dlpy.applications import YoloV2
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=4)
        model.print_summary()

    def test_yolov2_4(self):
        from dlpy.applications import YoloV2
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=1)
        model.print_summary()

    def test_yolov2_multi(self):
        from dlpy.applications import YoloV2_MultiSize
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2_MultiSize(self.s, anchors, predictions_per_grid=2)
        model.print_summary()

    def test_yolov2_multi_2(self):
        from dlpy.applications import YoloV2_MultiSize
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = YoloV2_MultiSize(self.s, anchors, predictions_per_grid=2, max_label_per_image=3, max_boxes=1)
        model.print_summary()

    def test_yolov2_tiny(self):
        from dlpy.applications import Tiny_YoloV2
        anchors=[]
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        anchors.append(1)
        model = Tiny_YoloV2(self.s, anchors, predictions_per_grid=2)
        model.print_summary()

    def test_yolov2_tiny_2(self):
        from dlpy.applications import Tiny_YoloV2
        anchors=[]
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
                            pre_trained_weights_file=self.data_dir+'InceptionV3_weights.kerasmodel.h5',
                            include_top=True)
        model.print_summary()

    def test_inceptionv3_3(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

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
                            pre_trained_weights_file=self.data_dir+'InceptionV3_weights.kerasmodel.h5',
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
