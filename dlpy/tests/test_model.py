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
#import onnx
import swat
import swat.utils.testing as tm
from swat.cas.table import CASTable
from dlpy.model import Model, Optimizer, AdamSolver, Sequence
from dlpy.sequential import Sequential
from dlpy.timeseries import TimeseriesTable
from dlpy.layers import (InputLayer, Conv2d, Pooling, Dense, OutputLayer,
                         Recurrent, Keypoints, BN, Res, Concat, Reshape)
from dlpy.utils import caslibify
from dlpy.applications import Tiny_YoloV2
import unittest


class TestModel(unittest.TestCase):
    '''
    Please locate the images.sashdat file under the datasources to the DLPY_DATA_DIR.
    '''
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

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            cls.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if cls.data_dir_local.endswith(cls.server_sep):
                cls.data_dir_local = cls.data_dir_local[:-1]
            cls.data_dir_local += cls.server_sep

    def test_model1(self):

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', lr=0.001)
        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertTrue(r.severity <= 1)

    def test_model2(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_')
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data='eee')
        self.assertTrue(r2.severity == 0)

    def test_model3(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_')
        self.assertTrue(r.severity == 0)

        r1 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=3)
        self.assertTrue(r1.severity == 0)

        r2 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=2)
        self.assertTrue(r2.severity == 0)

        r3 = model1.predict(data='eee')
        self.assertTrue(r3.severity == 0)

    def test_model4(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_')
        self.assertTrue(r.severity == 0)

        r2 = model1.evaluate(data='eee')
        self.assertTrue(r2.severity == 0)

    def test_model5(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_')
        self.assertTrue(r.severity == 0)

        r1 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=3)
        self.assertTrue(r1.severity == 0)

        r2 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=2)
        self.assertTrue(r2.severity == 0)

        r3 = model1.evaluate(data='eee')
        self.assertTrue(r3.severity == 0)

    def test_model6(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

    def test_model7(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data='eee', use_best_weights=True)
        self.assertTrue(r2.severity == 0)

    def test_model8(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data='eee')
        self.assertTrue(r2.severity == 0)

    def test_model9(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r2 = model1.evaluate(data='eee', use_best_weights=True)
        self.assertTrue(r2.severity == 0)

    def test_model10(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r2 = model1.evaluate(data='eee')
        self.assertTrue(r2.severity == 0)

        model1.save_to_table(self.data_dir)

    def test_model11(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r1 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=3)
        self.assertTrue(r1.severity == 0)

        r2 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=2)
        self.assertTrue(r2.severity == 0)

        r3 = model1.evaluate(data='eee', use_best_weights=True)
        self.assertTrue(r3.severity == 0)

    def test_model12(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)
        self.assertTrue(r.severity == 0)

        r1 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=3)
        self.assertTrue(r1.severity == 0)

        r2 = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=2, save_best_weights=True)
        self.assertTrue(r2.severity == 0)

        r3 = model1.predict(data='eee', use_best_weights=True)
        self.assertTrue(r3.severity == 0)

    def test_model13(self):
        model = Sequential(self.s, model_table='simple_cnn')
        model.add(InputLayer(3, 224, 224))
        model.add(Conv2d(2, 3))
        model.add(Pooling(2))
        model.add(Dense(4))
        model.add(OutputLayer(n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        model.save_to_table(self.data_dir)

    def test_model13a(self):
        model = Sequential(self.s, model_table='simple_cnn')
        model.add(InputLayer(3, 224, 224))
        model.add(Conv2d(2, 3))
        model.add(Pooling(2))
        model.add(Dense(4))
        model.add(OutputLayer(n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        model.save_to_table(self.data_dir)

    def test_model13b(self):
        model = Sequential(self.s, model_table='simple_cnn')
        model.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model.add(layer=OutputLayer(n=10, full_connect=False))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (0, 0))

        model1 = Sequential(self.s, model_table='simple_cnn')
        model1.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model1.add(layer=OutputLayer(n=10, full_connect=True))
        self.assertTrue(model1.summary.loc[1, 'Number of Parameters'] == (1000, 10))

        model2 = Sequential(self.s, model_table='Simple_CNN')
        model2.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model2.add(layer=OutputLayer(n=10, full_connect=True, include_bias=False))
        self.assertTrue(model2.summary.loc[1, 'Number of Parameters'] == (1000, 0))

        model3 = Sequential(self.s, model_table='Simple_CNN')
        model3.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model3.add(layer=Conv2d(4, 3))
        model3.add(layer=OutputLayer(n=10))
        self.assertTrue(model3.summary.loc[2, 'Number of Parameters'] == (4000, 10))

        model4 = Sequential(self.s, model_table='Simple_CNN')
        model4.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model4.add(layer=Conv2d(4, 3))
        model4.add(layer=OutputLayer(n=10, full_connect=False))
        self.assertTrue(model4.summary.loc[2, 'Number of Parameters'] == (0, 0))

    def test_model14(self):
        model = Sequential(self.s, model_table='Simple_CNN')
        model.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model.add(layer=OutputLayer())
        model.summary

    def test_model15(self):
        model = Sequential(self.s, model_table='Simple_CNN')
        model.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model.add(layer=Keypoints())
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (0, 0))

    def test_model16(self):
        model = Sequential(self.s, model_table='Simple_CNN')
        model.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model.add(layer=Keypoints(n=10, include_bias=False))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (1000, 0))

    def test_model16(self):
        model = Sequential(self.s, model_table='Simple_CNN')
        model.add(layer=InputLayer(n_channels=1, height=10, width=10))
        model.add(layer=Keypoints(n=10))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (1000, 10))

    def test_model18(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.save_weights_csv(self.data_dir)

    def test_model19(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.deploy(self.data_dir, output_format='onnx')

    def test_model20(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.save_weights_csv(self.data_dir)
        weights_path = os.path.join(self.data_dir, 'Simple_CNN1_weights.csv')
        model1.deploy(self.data_dir_local, output_format='onnx', model_weights=weights_path)

    def test_model21(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        pool1 = Pooling(2)
        model1.add(pool1)
        conv1 = Conv2d(1, 7, src_layers=[pool1])
        conv2 = Conv2d(1, 7, src_layers=[pool1])
        model1.add(conv1)
        model1.add(conv2)
        model1.add(Concat(act='identity', src_layers=[conv1, conv2]))
        model1.add(Pooling(2))
        model1.add(Dense(2))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.deploy(self.data_dir, output_format='onnx')

    def test_model22(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        pool1 = Pooling(2)
        model1.add(pool1)
        conv1 = Conv2d(1, 1, act='identity', src_layers=[pool1])
        model1.add(conv1)
        model1.add(Res(act='relu', src_layers=[conv1, pool1]))
        model1.add(Pooling(2))
        model1.add(Dense(2))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.deploy(self.data_dir, output_format='onnx')

    def test_model22_1(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")
        from onnx import numpy_helper
        import numpy as np

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7, act='identity', include_bias=False))
        model1.add(Reshape(height=448, width=448, depth=2))
        model1.add(Dense(2))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.deploy(self.data_dir_local, output_format='onnx')

        model_path = os.path.join(self.data_dir_local, 'Simple_CNN1.onnx')
        m = onnx.load(model_path)
        self.assertEqual(m.graph.node[1].op_type, 'Reshape')
        init = numpy_helper.to_array(m.graph.initializer[1])
        self.assertTrue(np.array_equal(init, [ -1,  2, 448, 448]))

    def test_model23(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7, act='identity', include_bias=False))
        model1.add(BN(act='relu'))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7, act='identity', include_bias=False))
        model1.add(BN(act='relu'))
        model1.add(Pooling(2))
        model1.add(Dense(2))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        model1.deploy(self.data_dir, output_format='onnx')

    def test_model24(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'model.onnx'))
        model1 = Model.from_onnx_model(self.s, m)
        model1.print_summary()

    def test_model25(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'model.onnx'))
        model1 = Model.from_onnx_model(self.s, m, offsets=[1, 1, 1,], scale=2, std='std')
        model1.print_summary()

    def test_model26(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'Simple_CNN1.onnx'))
        model1 = Model.from_onnx_model(self.s, m, offsets=[1, 1, 1,], scale=2, std='std')
        model1.print_summary()

    def test_model27(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'pytorch_net1.onnx'))
        model1 = Model.from_onnx_model(self.s, m, offsets=[1, 1, 1,], scale=2, std='std')
        model1.print_summary()

    def test_model28(self):
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'pytorch_net2.onnx'))
        model1 = Model.from_onnx_model(self.s, m, offsets=[1, 1, 1,], scale=2, std='std')
        model1.print_summary()

    def test_evaluate_obj_det(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path = self.data_dir + 'evaluate_obj_det_det.sashdat', task = 'load')

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'evaluate_obj_det_det', 'replace': True},
                               path = path)

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'evaluate_obj_det_gt', 'replace': True},
                               path = 'evaluate_obj_det_gt.sashdat')
        yolo_anchors = (5.9838598901098905,
                        3.4326923076923075,
                        2.184993862520458,
                        1.9841448445171848,
                        1.0261752136752136,
                        1.2277777777777779)
        yolo_model = Tiny_YoloV2(self.s, grid_number = 17, scale = 1.0 / 255,
                                 n_classes = 1, height = 544, width = 544,
                                 predictions_per_grid = 3,
                                 anchors = yolo_anchors,
                                 max_boxes = 100,
                                 coord_type = 'yolo',
                                 max_label_per_image = 100,
                                 class_scale = 1.0,
                                 coord_scale = 2.0,
                                 prediction_not_a_object_scale = 1,
                                 object_scale = 5,
                                 detection_threshold = 0.05,
                                 iou_threshold = 0.2)

        metrics = yolo_model.evaluate_object_detection(ground_truth = 'evaluate_obj_det_gt', coord_type = 'yolo',
                                                       detection_data = 'evaluate_obj_det_det', iou_thresholds=0.5)

    def test_model29(self):
        # test specifying output layer in Model.from_onnx_model
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'Simple_CNN1.onnx'))
        output_layer = OutputLayer(n=100)
        model1 = Model.from_onnx_model(conn=self.s,
                                       onnx_model=m,
                                       offsets=[1, 1, 1,],
                                       scale=2,
                                       std='std',
                                       output_layer=output_layer)

        self.assertTrue(model1.layers[-1].config['n'] == 100)

    def test_model30(self):
        # test specifying output layer in Model.from_onnx_model
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, "onnx not found in the libraries")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in "
                                             "the environment variables")

        m = onnx.load(os.path.join(self.data_dir_local, 'Simple_CNN1.onnx'))
        output_layer = OutputLayer(name='test_output', n=50)
        model1 = Model.from_onnx_model(conn=self.s,
                                       onnx_model=m,
                                       offsets=[1, 1, 1,],
                                       scale=2,
                                       std='std',
                                       output_layer=output_layer)

        self.assertTrue(model1.layers[-1].name == 'test_output')
        self.assertTrue(model1.layers[-1].config['n'] == 50)
        
    def test_model_forecast1(self):
        
        import datetime
        try:
            import pandas as pd
        except:
            unittest.TestCase.skipTest(self, "pandas not found in the libraries") 
        import numpy as np
            
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.table1 = TimeseriesTable.from_localfile(self.s, filename1, importoptions=importoptions1)
        self.table1.timeseries_formatting(timeid='datetime',
                                  timeseries=['series', 'covar'],
                                  timeid_informat='ANYDTDTM19.',
                                  timeid_format='DATETIME19.')
        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])
        self.table1.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series'],
                                         missing_handling='drop')
        
        valid_start = datetime.date(2015, 1, 4)
        test_start = datetime.date(2015, 1, 7)
        
        traintbl, validtbl, testtbl = self.table1.timeseries_partition(
                validation_start=valid_start, testing_start=test_start)
        
        model1 = Sequential(self.s, model_table='lstm_rnn')
        model1.add(InputLayer(std='STD'))
        model1.add(Recurrent(rnn_type='LSTM', output_type='encoding', n=15, reversed_=False))
        model1.add(OutputLayer(act='IDENTITY'))
        
        optimizer = Optimizer(algorithm=AdamSolver(learning_rate=0.01), mini_batch_size=32, 
                              seed=1234, max_epochs=10)                    
        seq_spec  = Sequence(**traintbl.sequence_opt)
        result = model1.fit(traintbl, valid_table=validtbl, optimizer=optimizer, 
                            sequence=seq_spec, **traintbl.inputs_target)
        
        self.assertTrue(result.severity == 0)
        
        resulttbl1 = model1.forecast(horizon=1)
        self.assertTrue(isinstance(resulttbl1, CASTable))
        self.assertTrue(resulttbl1.shape[0]==15)
        
        local_resulttbl1 = resulttbl1.to_frame()
        unique_time = local_resulttbl1.datetime.unique()
        self.assertTrue(len(unique_time)==1)
        self.assertTrue(pd.Timestamp(unique_time[0])==datetime.datetime(2015,1,7))

        resulttbl2 = model1.forecast(horizon=3)
        self.assertTrue(isinstance(resulttbl2, CASTable))
        self.assertTrue(resulttbl2.shape[0]==45)
        
        local_resulttbl2 = resulttbl2.to_frame()
        local_resulttbl2.sort_values(by=['id1var', 'id2var', 'datetime'], inplace=True)
        unique_time = local_resulttbl2.datetime.unique()
        self.assertTrue(len(unique_time)==3)
        for i in range(3):
            self.assertTrue(pd.Timestamp(unique_time[i])==datetime.datetime(2015,1,7+i))
        
        series_lag1 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag1'].values
                                           
        series_lag2 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag2'].values
        
        DL_Pred = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             '_DL_Pred_'].values
                                       
        self.assertTrue(np.array_equal(series_lag1[1:3], DL_Pred[0:2]))
        self.assertTrue(series_lag2[2]==DL_Pred[0])        

    def test_model_forecast2(self):
        
        import datetime
        try:
            import pandas as pd
        except:
            unittest.TestCase.skipTest(self, "pandas not found in the libraries") 
        import numpy as np
            
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.table2 = TimeseriesTable.from_localfile(self.s, filename1, importoptions=importoptions1)
        self.table2.timeseries_formatting(timeid='datetime',
                                  timeseries=['series', 'covar'],
                                  timeid_informat='ANYDTDTM19.',
                                  timeid_format='DATETIME19.')
        self.table2.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])
        self.table2.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')
        
        valid_start = datetime.date(2015, 1, 4)
        test_start = datetime.date(2015, 1, 7)
        
        traintbl, validtbl, testtbl = self.table2.timeseries_partition(
                validation_start=valid_start, testing_start=test_start)
        
        model1 = Sequential(self.s, model_table='lstm_rnn')
        model1.add(InputLayer(std='STD'))
        model1.add(Recurrent(rnn_type='LSTM', output_type='encoding', n=15, reversed_=False))
        model1.add(OutputLayer(act='IDENTITY'))
        
        optimizer = Optimizer(algorithm=AdamSolver(learning_rate=0.01), mini_batch_size=32, 
                              seed=1234, max_epochs=10)                    
        seq_spec  = Sequence(**traintbl.sequence_opt)
        result = model1.fit(traintbl, valid_table=validtbl, optimizer=optimizer, 
                            sequence=seq_spec, **traintbl.inputs_target)
        
        self.assertTrue(result.severity == 0)
        
        resulttbl1 = model1.forecast(testtbl, horizon=1)
        self.assertTrue(isinstance(resulttbl1, CASTable))
        self.assertTrue(resulttbl1.shape[0]==testtbl.shape[0])
        
        local_resulttbl1 = resulttbl1.to_frame()
        unique_time = local_resulttbl1.datetime.unique()
        self.assertTrue(len(unique_time)==4)
        for i in range(4):
            self.assertTrue(pd.Timestamp(unique_time[i])==datetime.datetime(2015,1,7+i))

        resulttbl2 = model1.forecast(testtbl, horizon=3)
        self.assertTrue(isinstance(resulttbl2, CASTable))
        self.assertTrue(resulttbl2.shape[0]==45)
        
        local_resulttbl2 = resulttbl2.to_frame()
        local_resulttbl2.sort_values(by=['id1var', 'id2var', 'datetime'], inplace=True)
        unique_time = local_resulttbl2.datetime.unique()
        self.assertTrue(len(unique_time)==3)
        for i in range(3):
            self.assertTrue(pd.Timestamp(unique_time[i])==datetime.datetime(2015,1,7+i))
        
        series_lag1 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag1'].values
                                           
        series_lag2 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag2'].values
        
        DL_Pred = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             '_DL_Pred_'].values
                                       
        self.assertTrue(np.array_equal(series_lag1[1:3], DL_Pred[0:2]))
        self.assertTrue(series_lag2[2]==DL_Pred[0])        

    def test_model_forecast3(self):
        
        import datetime
        try:
            import pandas as pd
        except:
            unittest.TestCase.skipTest(self, "pandas not found in the libraries") 
        import numpy as np
            
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.table3 = TimeseriesTable.from_localfile(self.s, filename1, importoptions=importoptions1)
        self.table3.timeseries_formatting(timeid='datetime',
                                  timeseries=['series', 'covar'],
                                  timeid_informat='ANYDTDTM19.',
                                  timeid_format='DATETIME19.')
        self.table3.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])
        self.table3.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')
        
        valid_start = datetime.date(2015, 1, 4)
        test_start = datetime.date(2015, 1, 7)
        
        traintbl, validtbl, testtbl = self.table3.timeseries_partition(
                validation_start=valid_start, testing_start=test_start)
        
        sascode = '''
        data {};
        set {};
        drop series_lag1;
        run;
        '''.format(validtbl.name, validtbl.name)
        
        self.s.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
        
        sascode = '''
        data {};
        set {};
        drop series_lag1;
        run;
        '''.format(testtbl.name, testtbl.name)
        
        self.s.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
        
        model1 = Sequential(self.s, model_table='lstm_rnn')
        model1.add(InputLayer(std='STD'))
        model1.add(Recurrent(rnn_type='LSTM', output_type='encoding', n=15, reversed_=False))
        model1.add(OutputLayer(act='IDENTITY'))
        
        optimizer = Optimizer(algorithm=AdamSolver(learning_rate=0.01), mini_batch_size=32, 
                              seed=1234, max_epochs=10)                    
        seq_spec  = Sequence(**traintbl.sequence_opt)
        result = model1.fit(traintbl, optimizer=optimizer, 
                            sequence=seq_spec, **traintbl.inputs_target)
        
        self.assertTrue(result.severity == 0)
        
        resulttbl1 = model1.forecast(validtbl, horizon=1)
        self.assertTrue(isinstance(resulttbl1, CASTable))
        self.assertTrue(resulttbl1.shape[0]==15)
        
        local_resulttbl1 = resulttbl1.to_frame()
        unique_time = local_resulttbl1.datetime.unique()
        self.assertTrue(len(unique_time)==1)
        self.assertTrue(pd.Timestamp(unique_time[0])==datetime.datetime(2015,1,4))

        resulttbl2 = model1.forecast(validtbl, horizon=3)
        self.assertTrue(isinstance(resulttbl2, CASTable))
        self.assertTrue(resulttbl2.shape[0]==45)
        
        local_resulttbl2 = resulttbl2.to_frame()
        local_resulttbl2.sort_values(by=['id1var', 'id2var', 'datetime'], inplace=True)
        unique_time = local_resulttbl2.datetime.unique()
        self.assertTrue(len(unique_time)==3)
        for i in range(3):
            self.assertTrue(pd.Timestamp(unique_time[i])==datetime.datetime(2015,1,4+i))
        
        series_lag1 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag1'].values
                                           
        series_lag2 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag2'].values
        
        DL_Pred = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             '_DL_Pred_'].values
                                       
        self.assertTrue(np.array_equal(series_lag1[1:3], DL_Pred[0:2]))
        self.assertTrue(series_lag2[2]==DL_Pred[0]) 
        
        with self.assertRaises(RuntimeError):
            resulttbl3 = model1.forecast(testtbl, horizon=3)
        
        
    def test_imagescaler1(self):
        # test import model with imagescaler
        try:
            import onnx
            from onnx import helper, TensorProto
        except:
            unittest.TestCase.skipTest(self, 'onnx not found')

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, 'DLPY_DATA_DIR_LOCAL is not set in '
                                             'the environment variables')

        import numpy as np
        n1 = helper.make_node('ImageScaler',
                              ['X'],
                              ['X1'],
                              bias=[0., 0., 0.],
                              scale=1.)
        n2 = helper.make_node('Conv',
                              inputs=['X1', 'W1'],
                              outputs=['X2'],
                              kernel_shape=[3, 3],
                              pads=[0, 0, 0, 0])
        n3 = helper.make_node('MatMul',
                              inputs=['X2', 'W2'],
                              outputs=['X3'])

        W1 = np.ones((3, 3, 3)).astype(np.float32)
        W2 = np.ones((9, 2)).astype(np.float32)

        graph_def = helper.make_graph(
            [n1, n2, n3],
            name='test',
            inputs=[
                helper.make_tensor_value_info('X',
                                              TensorProto.FLOAT,
                                              [1, 3, 10, 10]),
                helper.make_tensor_value_info('W1',
                                              TensorProto.FLOAT,
                                              [3, 3, 3]),
                helper.make_tensor_value_info('W2',
                                              TensorProto.FLOAT,
                                              [9, 2])],
            outputs=[
                helper.make_tensor_value_info('X3',
                                              TensorProto.FLOAT,
                                              [1, 2])],
            initializer=[
                helper.make_tensor('W1',
                                   TensorProto.FLOAT,
                                   [3, 3, 3],
                                   W1.flatten().astype(np.float32)),
                helper.make_tensor('W2',
                                   TensorProto.FLOAT,
                                   [9, 2],
                                   W2.flatten().astype(np.float32))])
        onnx_model =  helper.make_model(graph_def)

        model1 = Model.from_onnx_model(self.s, onnx_model)

        l1 = model1.layers[0]
        self.assertTrue(l1.type == 'input')
        self.assertTrue(l1.config['offsets'] == [0., 0., 0.])
        self.assertTrue(l1.config['scale'] == 1.)

    def test_imagescaler2(self):
        # test export model with imagescaler
        try:
            import onnx
        except:
            unittest.TestCase.skipTest(self, 'onnx not found')

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, 'DLPY_DATA_DIR_LOCAL is not set in '
                                             'the environment variables')

        model1 = Sequential(self.s, model_table='imagescaler2')
        model1.add(InputLayer(n_channels=3,
                              width=224,
                              height=224,
                              scale=1/255.,
                              offsets=[0.1, 0.2, 0.3]))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(OutputLayer(act='softmax', n=2))

        caslib, path = caslibify(self.s,
                                 path=self.data_dir+'images.sashdat',
                                 task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)
        r = model1.fit(data='eee', inputs='_image_', target='_label_', max_epochs=1)
        self.assertTrue(r.severity == 0)

        from dlpy.model_conversion.write_onnx_model import sas_to_onnx
        onnx_model = sas_to_onnx(model1.layers,
                                 self.s.CASTable('imagescaler2'),
                                 self.s.CASTable('imagescaler2_weights'))

        self.assertAlmostEqual(onnx_model.graph.node[0].attribute[0].floats[0], 0.1)
        self.assertAlmostEqual(onnx_model.graph.node[0].attribute[0].floats[1], 0.2)
        self.assertAlmostEqual(onnx_model.graph.node[0].attribute[0].floats[2], 0.3)
        self.assertAlmostEqual(onnx_model.graph.node[0].attribute[1].f, 1/255.)

    def test_load_reshape_detection(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        yolo_model = Model(self.s)
        yolo_model.load(self.data_dir + 'YOLOV2_MULTISIZE.sashdat')
        model_df = self.s.fetch(table = dict(name = yolo_model.model_name,
                                             where = '_DLKey0_ eq "detection1" or _DLKey0_ eq "reshape1"'), to = 50).Fetch
        anchors_5 = model_df['_DLNumVal_'][model_df['_DLKey1_'] == 'detectionopts.anchors.8'].tolist()[0]
        self.assertAlmostEqual(anchors_5, 1.0907, 4)
        depth = model_df['_DLNumVal_'][model_df['_DLKey1_'] == 'reshapeopts.depth'].tolist()[0]
        self.assertEqual(depth, 256)

    def test_plot_ticks(self):

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', lr=0.001, max_epochs=5)
        
        # Test default tick_frequency value of 1
        ax = model1.plot_training_history()
        self.assertEqual(len(ax.xaxis.majorTicks), model1.n_epochs)

        # Test even
        tick_frequency = 2
        ax = model1.plot_training_history(tick_frequency=tick_frequency)
        self.assertEqual(len(ax.xaxis.majorTicks), model1.n_epochs // tick_frequency + 1)

        # Test odd
        tick_frequency = 3
        ax = model1.plot_training_history(tick_frequency=tick_frequency)
        self.assertEqual(len(ax.xaxis.majorTicks), model1.n_epochs // tick_frequency + 1)

        # Test max
        tick_frequency = model1.n_epochs
        ax = model1.plot_training_history(tick_frequency=tick_frequency)
        self.assertEqual(len(ax.xaxis.majorTicks), model1.n_epochs // tick_frequency + 1)
        
        # Test 0 
        tick_frequency = 0
        ax = model1.plot_training_history(tick_frequency=tick_frequency)
        self.assertEqual(len(ax.xaxis.majorTicks), model1.n_epochs)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
