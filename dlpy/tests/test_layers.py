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
import json
import os

from dlpy.layers import InputLayer, Conv2d, Pooling, Dense, Recurrent, BN, Res, Proj, OutputLayer, \
                        Keypoints, Detection, Scale, Reshape, Conv2DTranspose, GroupConv2d, GlobalAveragePooling2D, \
                        FastRCNN, ROIPooling, RegionProposal
from dlpy.blocks import ResBlock, ResBlockBN, ResBlock_Caffe, DenseNetBlock, Bidirectional
from dlpy.utils import DLPyError, get_mapping_dict
from dlpy import __dev__


class TestLayers(unittest.TestCase):

    sample_syntax = ''

    @classmethod
    def setUpClass(cls):
        filename = os.path.join('datasources', 'sample_syntax_for_test.json')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)

        with open(full_filename) as f:
            cls.sample_syntax = json.load(f)

    def test_input_layer1(self):
        dict1 = InputLayer(name='input1').to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer1'] == dict1)

    def test_input_layer2(self):
        dict1 = InputLayer(name='data').to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer2'] == dict1)

    def test_input_layer3(self):
        dict1 = InputLayer(name='data', n_channels=1).to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer3'] == dict1)

    def test_input_layer4(self):
        dict1 = InputLayer(name='data', n_channels=3).to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer4'] == dict1)

    def test_input_layer5(self):
        dict1 = InputLayer(name='data', n_channels=3, width=100).to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer5'] == dict1)

    def test_input_layer6(self):
        dict1 = InputLayer(name='data', n_channels=3, height=100).to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer6'] == dict1)

    def test_input_layer7(self):
        dict1 = InputLayer(name='data', n_channels=3, width=100, height=150).to_model_params()
        self.assertTrue(self.sample_syntax['inputlayer7'] == dict1)

    def test_input_layer8(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                InputLayer(not_a_parameter=1)

    def test_conv2d_layer1(self):
        dict1 = Conv2d(name='convo1', n_filters=10, act='relu',
                       src_layers=[InputLayer(name='input1')]).to_model_params()
        self.assertTrue(self.sample_syntax['convo1'] == dict1)

    def test_conv2d_layer2(self):
        dict1 = Conv2d(n_filters=32, width=5, height=7, name='convo2',
                       src_layers=[InputLayer(name='input1')]).to_model_params()
        self.assertTrue(self.sample_syntax['convo2'] == dict1)

    def test_conv2d_layer3(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Conv2d(n_filters=3, not_a_parameter=1)

    def test_conv2d_layer_name_format(self):
        if __dev__:
            dict1 = Conv2d(n_filters=32, width=5, height=7, name='convo2', includeBias=False)
            bias = dict1.num_bias
            self.assertTrue(bias == 0)

    def test_conv2d_layer_name_format2(self):
        if __dev__:
            dict1 = Conv2d(n_filters=32, width=5, height=7, name='convo2', include_bias=False)
            bias = dict1.num_bias
            self.assertTrue(bias == 0)

    def test_conv2d_layer_name_conflict(self):
        if __dev__:
            dict1 = Conv2d(n_filters=32, width=5, height=7, name='convo2',
                           stride_horizontal = 1, strideHorizontal=10,
                           include_bias=False, includeBias=True)
            bias = dict1.num_bias
            self.assertTrue(bias == 32)

    def test_pool_layer1(self):
        dict1 = Pooling(name='pool1', src_layers=[ Conv2d(n_filters=3, name='conv')]).to_model_params()
        self.assertTrue(self.sample_syntax['pool1'] == dict1)

    def test_pool_layer2(self):
        dict1 = Pooling(name='pool2', width=3, height=2,
                        src_layers=[Conv2d(n_filters=3, name='conv')]).to_model_params()
        self.assertTrue(self.sample_syntax['pool2'] == dict1)

    def test_pool_layer3(self):
        dict1 = Pooling(name='pool3', stride=1, width=3, height=2, pool='mean',
                        src_layers=[ Conv2d(n_filters=3, name='conv')]).to_model_params()
        self.assertTrue(self.sample_syntax['pool3'] == dict1)

    def test_pool_layer4(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Pooling(not_a_parameter=1)

    def test_dense_layer1(self):
        dict1 = Dense(name='dense', n=10, src_layers=[ Pooling(name='pool')]).to_model_params()
        self.assertTrue(self.sample_syntax['fc1'] == dict1)

    def test_dense_layer2(self):
        dict1 = Dense(name='dense', n=10000, init='xavier', dropout=0.2, include_bias=False,
                      src_layers=[Pooling(name='pool')] ).to_model_params()
        self.assertTrue(self.sample_syntax['fc2'] == dict1)

    def test_dense_layer3(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Dense(n=3, not_a_parameter=1)

    def test_rnn_layer1(self):
        dict1 = Recurrent(name='rnn', n=100, rnn_type='gru', init='xavier', output_type='samelength', reversed_=True,
                          src_layers=[InputLayer(name='data')]).to_model_params()
        print(dict1)
        self.assertTrue(self.sample_syntax['rnn1'] == dict1)

    def test_bn_layer1(self):
        dict1 = BN(name='bn', src_layers=[Conv2d(name='conv', n_filters=32)]).to_model_params()
        self.assertTrue(self.sample_syntax['bn1'] == dict1)

    def test_bn_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                BN(not_a_parameter=1)

    def test_res_layer1(self):
        if not __dev__:
            dict1 = Res(name='res', src_layers=[Conv2d(name='conv', n_filters=32)]).to_model_params()
            self.assertTrue(self.sample_syntax['res1'] == dict1)

    def test_res_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Res(not_a_parameter=1)

    def test_proj_layer1(self):
        dict1 = Proj(name='proj', embedding_size=100, alphabet_size=250,
                     src_layers=[InputLayer(name='data')]).to_model_params()
        self.assertTrue(self.sample_syntax['proj1'] == dict1)

    def test_proj_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Proj(embedding_size=1, alphabet_size=2, not_a_parameter=1)

    def test_output_layer1(self):
        dict1 = OutputLayer(name='output', n=100, src_layers=[Pooling(name='pool')]).to_model_params()
        self.assertTrue(self.sample_syntax['output1'] == dict1)

    def test_output_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                OutputLayer(not_a_parameter=1)

    def test_keypoints_layer1(self):
        dict1 = Keypoints(name='keypoints', n=100, src_layers=[Dense(name='fc', n=100)]).to_model_params()
        self.assertTrue(self.sample_syntax['keypoints1'] == dict1)

    def test_keypoints_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Keypoints(not_a_parameter=1)

    def test_detection_layer1(self):
        dict1 = Detection(name='detection', predictions_per_grid=7, iou_threshold=0.2, detection_threshold=0.2,
                          src_layers=[Pooling(name='pool')]).to_model_params()
        self.assertTrue(self.sample_syntax['detection1'] == dict1)

    def test_detection_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Detection(not_a_parameter=1)

    def test_scale_layer1(self):
        dict1 = Scale(name='scale', src_layers=[Pooling(name='pool')]).to_model_params()
        self.assertTrue(self.sample_syntax['scale1'] == dict1)

    def test_scale_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Scale(not_a_parameter=1)

    def test_reshape_layer1(self):
        dict1 = Reshape(name='reshape', width=1, height=2, depth=3,
                        src_layers=[Dense(name='fc', n=100)]).to_model_params()
        self.assertTrue(self.sample_syntax['reshape1'] == dict1)

    def test_reshape_layer2(self):
        if not __dev__:
            with self.assertRaises(DLPyError):
                Reshape(not_a_parameter=1)

    def test_summary_function(self):
        ol = OutputLayer(name='output', n=100)
        self.assertTrue(ol.summary['Output Size'][0] == 100)

    def test_formant_name_function(self):
        ol = Reshape()
        ol.format_name(block_num=1, local_count=7)
        self.assertTrue(ol.name == 'Reshape1_7')

    def test_residual_block1(self):
        list1 = ResBlock().compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['resblock1'] == list1)

    def test_residual_block11(self):
        rb1 = ResBlock(strides=1)
        self.assertTrue(len(rb1.strides) == 2)

        rb2 = ResBlock(strides=(1, 1))
        self.assertTrue(len(rb2.strides) == 2)

        rb3 = ResBlock(strides=(1))
        self.assertTrue(len(rb3.strides) == 2)

    def test_residual_block2(self):
        list1 = ResBlockBN().compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['resblock2'] == list1)

    def test_residual_block3(self):
        list1 = ResBlock_Caffe().compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['resblock3'] == list1)

    def test_residual_block4(self):
        try:
            ResBlock(kernel_sizes=3, n_filters=(16, 16), strides=(1, 1, 1))
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

    def test_residual_block5(self):
        try:
            ResBlock(kernel_sizes=(3, 3, 3), n_filters=(16, 16))
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

    def test_residual_block6(self):
        list1 = ResBlock_Caffe(conv_short_cut=True).compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['resblock4'] == list1)

    def test_residual_block7(self):
        list1 = ResBlock_Caffe(batch_norm_first=True).compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['resblock5'] == list1)

    def test_densenet_block1(self):
        list1 = DenseNetBlock(n_cells=1).compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['denseblock1'] == list1)

    def test_densenet_block2(self):
        list1 = DenseNetBlock().compile(src_layer=InputLayer(name='data'), block_num=1)
        self.assertTrue(self.sample_syntax['denseblock2'] == list1)

    def test_bidirectional_block1(self):
        list1 = Bidirectional(n=30).compile()
        self.assertTrue(self.sample_syntax['bidirectional1'] == list1)

    def test_bidirectional_block3(self):
        try:
            Bidirectional(n=[10, 20, 30], n_blocks=2).compile()
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

    def test_bidirectional_block4(self):
        list1 = Bidirectional(n=[10, 20, 30], n_blocks=3).compile()
        self.assertTrue(self.sample_syntax['bidirectional3'] == list1)

    def test_transpose_conv1(self):
        trans1 = Conv2DTranspose(n_filters=30, stride_horizontal = 2)
        self.assertTrue(trans1.padding == (0, 0))
        self.assertTrue(trans1.stride == (1, 2))
        self.assertTrue(trans1.output_padding == (0, 0))

        trans2 = Conv2DTranspose(n_filters = 30, padding_width = 3, stride = 10, output_padding = 2)
        self.assertTrue(trans2.padding == (0, 3))
        self.assertTrue(trans2.stride == (10, 10))
        self.assertTrue(trans2.output_padding == (2, 2))

        trans3 = Conv2DTranspose(n_filters = 30, padding_width = 3, stride = 10, output_padding_width = 2)
        self.assertTrue(trans3.output_padding == (0, 2))

    def test_group_conv1(self):
        group_conv1 = GroupConv2d(n_filters=30, n_groups=3, stride_horizontal = 2)
        self.assertTrue(group_conv1.config['nGroups'] == 3)

    def test_global_pooling1(self):
        global_pooling = GlobalAveragePooling2D()
        self.assertTrue(global_pooling.config['width'] == 0)

    def test_region_proposal1(self):
        region_proposal = RegionProposal(base_anchor_size = 10, anchor_ratio = [0.1, 2.0, 3.0],
                                         anchor_scale = [1.2, 2.3, 3.4])
        self.assertTrue(self.sample_syntax['region_proposal1'] == region_proposal.config)

    def test_fast_rcnn(self):
        fast_rcnn = FastRCNN(class_number = 10, detection_threshold = 0.2, max_label_per_image = 60,
                             nms_iou_threshold = 0.3)
        self.assertTrue(self.sample_syntax['fast_rcnn1'] == fast_rcnn.config)

    def test_mapping_dict(self):
        mapping = get_mapping_dict()
        print(mapping['learningrate'])
