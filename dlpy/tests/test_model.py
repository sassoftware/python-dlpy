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
import swat
import swat.utils.testing as tm
from dlpy.sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, Pooling, Dense, OutputLayer, Keypoints
from dlpy.utils import caslibify
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

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
