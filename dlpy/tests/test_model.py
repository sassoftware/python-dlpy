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
from dlpy.images import ImageTable
from dlpy.splitting import two_way_split
from dlpy.model import *
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

    def test_model13(self):
        model = Sequential(self.s, model_table='simple_cnn')
        model.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model.add(layer = OutputLayer(n = 10, full_connect = False))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (0, 0))

        model1 = Sequential(self.s, model_table = 'simple_cnn')
        model1.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model1.add(layer = OutputLayer(n = 10, full_connect = True))
        self.assertTrue(model1.summary.loc[1, 'Number of Parameters'] == (1000, 10))

        model2 = Sequential(self.s, model_table = 'Simple_CNN')
        model2.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model2.add(layer = OutputLayer(n = 10, full_connect = True, include_bias = False))
        self.assertTrue(model2.summary.loc[1, 'Number of Parameters'] == (1000, 0))

        model3 = Sequential(self.s, model_table = 'Simple_CNN')
        model3.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model3.add(layer = Conv2d(4, 3))
        model3.add(layer = OutputLayer(n = 10))
        self.assertTrue(model3.summary.loc[2, 'Number of Parameters'] == (4000, 10))

        model4 = Sequential(self.s, model_table = 'Simple_CNN')
        model4.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model4.add(layer = Conv2d(4, 3))
        model4.add(layer = OutputLayer(n = 10, full_connect = False))
        self.assertTrue(model4.summary.loc[2, 'Number of Parameters'] == (0, 0))

    def test_model14(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model.add(layer = OutputLayer())
        model.summary

    def test_model15(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model.add(layer = Keypoints())
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (0, 0))

    def test_model16(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model.add(layer = Keypoints(n = 10, include_bias = False))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (1000, 0))

    def test_model16(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(layer = InputLayer(n_channels = 1, height = 10, width = 10))
        model.add(layer = Keypoints(n = 10))
        self.assertTrue(model.summary.loc[1, 'Number of Parameters'] == (1000, 10))

    def test_data_specs_dict(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [dict(type = 'IMAGE', layer = 'Input1', data = '_image_'),
                      dict(type = 'NUMERICNOMINAL', layer = 'Output1', data = '_label_')]
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')

    def test_data_specs_dict2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file', cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [dict(type = 'IMAGE', layer = 'Input1', data = 'img'),
                      dict(type = 'NUMERICNOMINAL', layer = 'Output1', data = 'label')]
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')

    def test_image_table(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        tr_img, te_img = two_way_split(my_images, test_rate = 20, stratify_by = my_images.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        model1.fit(data = tr_img,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')

    ''' bug here '''
    def test_image_table2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file', cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        model1.fit(data = tr_img,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')

    def test_data_specs(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = '_image_'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = '_label_')]
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')
        model1.fit(data = tr_img, target = '_label_', inputs = '_image_',
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        model1.evaluate(te_img)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')

    def test_data_specs2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file', cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = 'img'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = 'label')]
        model1.fit(data = tr_img,  # target = 'label', inputs = 'img',
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        model1.evaluate(te_img)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')

    def test_heat_map_analysis(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = '_image_'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = '_label_')]
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')
        model1.evaluate(te_img)
        filter_list = list(range(500))
        model1.heat_map_analysis(data = te_img, mask_width = 56, mask_height = 56,
                                 step_size = 8, max_display = 2, display=False)

        model1.heat_map_analysis(data = te_img, mask_width = 56, mask_height = 56, step_size = 8, max_display = 2,
                                 filter_column = te_img.id_col, filter_list = filter_list, display=False)

    '''train with datasteps and score img'''
    def test_heat_map_analysis2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file', cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = 'img'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = 'label')]
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')
        model1.evaluate(te_img)
        filter_list = list(range(500))
        model1.heat_map_analysis(data = te_img, mask_width = 56, mask_height = 56,
                                 step_size = 8, max_display = 2, display=False)

        model1.heat_map_analysis(data = te_img, mask_width = 56, mask_height = 56, step_size = 8, max_display = 2,
                                 filter_column = te_img.id_col, filter_list = filter_list, display=False)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')

    def test_load(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        model = Model(self.s)
        model_file = '/bigdisk/lax/dlpy/vgg16.sashdat'
        model.load(path = model_file)
        model.evaluate(te_img, target = '_label_', input = '_image_')

    def test_get_feature_maps(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = '_image_'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = '_label_')]
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        model1.evaluate(te_img)
        model1.get_feature_maps(data = te_img)

    def test_plot_evaluate_res_with_filter(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = '_image_'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = '_label_')]
        model1.fit(data = tr_img, target = '_label_', inputs = '_image_',
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')
        model1.evaluate(te_img)
        filter_list = list(range(100))
        model1.plot_evaluate_res(img_type = 'C', randomize = True, n_images = 2,
                                 filter_column = tr_img.id_col, filter_list = filter_list)

    def test_evaluate(self):
        img_path = '/bigdisk/lax/dlpy/Giraffe_Dolphin'
        my_images = ImageTable.load_files(self.s, path = img_path)
        my_images.resize(width = 224)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        tr_img, te_img = two_way_split(my_images, test_rate = 1, seed = 123)
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2))
        model1.fit(data = tr_img,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == '_image_' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == '_label_')
        model1.print_summary()
        model1.evaluate(te_img)

    def test_evaluate2(self):
        self.s.table.addcaslib(activeonadd = False,
                               datasource = {'srctype': 'path'},
                               name = 'dnfs',
                               path = '/bigdisk/lax/dlpy/',
                               subdirectories = False)
        self.s.table.loadTable(caslib = 'dnfs', path = 'imageTable_test.sashdat',
                               casout = dict(name = 'data', replace = True))
        test = ImageTable.from_table(tbl = self.s.CASTable('data'), image_col = 'img', id_col = 'id',
                                     filename_col = 'file', cls_cols = 'label')
        tr_img, te_img = two_way_split(test, test_rate = 20, stratify_by = test.cls_cols, seed = 123)
        model1 = Sequential(self.s, model_table = 'Simple_CNN')
        model1.add(InputLayer(3, 224, 224, offsets = tr_img.channel_means, name = 'Input1'))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2, name = 'Output1'))
        data_specs = [DataSpec(type_ = 'IMAGE', layer = 'Input1', data = 'img'),
                      DataSpec(type_ = 'NUMERICNOMINAL', layer = 'Output1', data = 'label')]
        te_img.as_patches(width = 200, height = 200, step_size = 24, output_width = 224, output_height = 224)
        model1.fit(data = tr_img,
                   data_specs = data_specs,
                   mini_batch_size = 2,
                   max_epochs = 2,
                   lr = 1E-4,
                   log_level = 2)
        self.assertTrue(model1.tasks[0] == 'classification_0' and len(model1.tasks) == 1)
        self.assertTrue(model1.inputs[0] == 'img' and len(model1.inputs) == 1)
        self.assertTrue(model1.targets['classification_0'] == 'label')
        model1.print_summary()
        model1.predict(te_img)
        model1.evaluate(te_img)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
