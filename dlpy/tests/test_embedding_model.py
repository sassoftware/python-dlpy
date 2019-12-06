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

from dlpy import Dense
from dlpy.applications import ResNet18_Caffe, MobileNetV1
from dlpy.embedding_model import EmbeddingModel


class TestImageEmbeddingModel(unittest.TestCase):
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

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            cls.local_dir = os.environ.get("DLPY_DATA_DIR_LOCAL")

        # the server path that points to DLPY_DATA_DIR_LOCAL
        if "DLPY_DATA_DIR_SERVER" in os.environ:
            cls.server_dir = os.environ.get("DLPY_DATA_DIR_SERVER")
            if cls.server_dir.endswith(cls.server_sep):
                cls.server_dir = cls.server_dir[:-1]
            cls.server_dir += cls.server_sep

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_embedding_model_siamese(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test default
        resnet18_model = ResNet18_Caffe(self.s,
                                        width=224,
                                        height=224,
                                        random_flip='HV',
                                        random_mutation='random'
                                        )
        branch = resnet18_model.to_functional_model(stop_layers=resnet18_model.layers[-1])
        model = EmbeddingModel.build_embedding_model(branch)
        res = model.print_summary()
        # print(res)
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 2)

        # test options
        embedding_layer = Dense(n=10)
        model1 = EmbeddingModel.build_embedding_model(branch, model_table='test',
                                                      embedding_model_type='siamese', margin=3.0,
                                                      embedding_layer=embedding_layer)
        res1 = model1.print_summary()
        # print(res1)
        self.assertEqual(res1[res1['Layer'].str.contains(model1.embedding_layer_name_prefix)].shape[0], 2)

        # test passing in a sequential model
        model2 = EmbeddingModel.build_embedding_model(resnet18_model)
        res2 = model2.print_summary()
        self.assertEqual(res2[res2['Layer'].str.contains(model2.embedding_layer_name_prefix)].shape[0], 2)

        model3 = EmbeddingModel.build_embedding_model(resnet18_model, model_table='test2',
                                                      embedding_model_type='siamese', margin=3.0,
                                                      embedding_layer=embedding_layer)
        res3 = model3.print_summary()
        self.assertEqual(res3[res3['Layer'].str.contains(model3.embedding_layer_name_prefix)].shape[0], 2)

    def test_embedding_model_siamese_1(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test passing in a functional model
        vgg16 = Model(self.s)
        vgg16.load(path=self.data_dir + 'vgg16.sashdat')
        model = EmbeddingModel.build_embedding_model(vgg16)
        res = model.print_summary()
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 2)

    def test_embedding_model_triplet(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test triplet
        resnet18_model = ResNet18_Caffe(self.s,
                                        width=224,
                                        height=224,
                                        random_flip='HV',
                                        random_mutation='random'
                                        )
        branch = resnet18_model.to_functional_model(stop_layers=resnet18_model.layers[-1])

        model = EmbeddingModel.build_embedding_model(branch, model_table='test',
                                                     embedding_model_type='triplet', margin=-3.0)
        res = model.print_summary()
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 3)

        # test embedding layer
        embedding_layer = Dense(n=10)
        model1 = EmbeddingModel.build_embedding_model(branch, model_table='test',
                                                      embedding_model_type='triplet', margin=-3.0,
                                                      embedding_layer=embedding_layer)
        res1 = model1.print_summary()
        # print(res1)
        self.assertEqual(res1[res1['Layer'].str.contains(model1.embedding_layer_name_prefix)].shape[0], 3)

        # test passing in a sequential model
        model2 = EmbeddingModel.build_embedding_model(resnet18_model,
                                                      embedding_model_type='triplet', margin=-3.0,
                                                      embedding_layer=embedding_layer)
        res2 = model2.print_summary()
        self.assertEqual(res2[res2['Layer'].str.contains(model2.embedding_layer_name_prefix)].shape[0], 3)

    def test_embedding_model_triplet_1(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test passing in a functional model
        vgg16 = Model(self.s)
        vgg16.load(path=self.data_dir + 'vgg16.sashdat')
        embedding_layer = Dense(n=10)
        model = EmbeddingModel.build_embedding_model(vgg16,
                                                     embedding_model_type='triplet', margin=-3.0,
                                                     embedding_layer=embedding_layer)
        res = model.print_summary()
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 3)

    def test_embedding_model_quartet(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test triplet
        resnet18_model = ResNet18_Caffe(self.s,
                                        width=224,
                                        height=224,
                                        random_flip='HV',
                                        random_mutation='random'
                                        )
        branch = resnet18_model.to_functional_model(stop_layers=resnet18_model.layers[-1])

        model = EmbeddingModel.build_embedding_model(branch, model_table='test',
                                                     embedding_model_type='quartet', margin=-3.0)
        res = model.print_summary()
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 4)

        # test embedding layer
        embedding_layer = Dense(n=10)
        model1 = EmbeddingModel.build_embedding_model(branch, model_table='test',
                                                      embedding_model_type='quartet', margin=-3.0,
                                                      embedding_layer=embedding_layer)
        res1 = model1.print_summary()
        # print(res1)
        self.assertEqual(res1[res1['Layer'].str.contains(model1.embedding_layer_name_prefix)].shape[0], 4)

        # test passing in a sequential model
        model2 = EmbeddingModel.build_embedding_model(resnet18_model,
                                                      embedding_model_type='quartet', margin=-3.0,
                                                      embedding_layer=embedding_layer)
        res2 = model2.print_summary()
        self.assertEqual(res2[res2['Layer'].str.contains(model2.embedding_layer_name_prefix)].shape[0], 4)

    def test_embedding_model_quartet_1(self):

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        # test passing in a functional model
        vgg16 = Model(self.s)
        vgg16.load(path=self.data_dir + 'vgg16.sashdat')
        embedding_layer = Dense(n=10)
        model = EmbeddingModel.build_embedding_model(vgg16,
                                                     embedding_model_type='quartet', margin=-3.0,
                                                     embedding_layer=embedding_layer)
        res = model.print_summary()
        self.assertEqual(res[res['Layer'].str.contains(model.embedding_layer_name_prefix)].shape[0], 4)

