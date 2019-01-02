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
import os
from dlpy.lr_scheduler import StepLR, FixedLR, ReduceLROnPlateau, CyclicLR
from dlpy.sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, Pooling, Dense, OutputLayer
from dlpy.utils import caslibify
from dlpy.model import VanillaSolver, Optimizer


class TestLRScheduler(unittest.TestCase):
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

        cls.s = swat.CAS('dlgrd009', 13319)
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

    def test_StepLR(self):
        lr_policy = StepLR(1.0, 1.2, 30)
        print(lr_policy)

    def test_CyclicLR(self):
        model1 = Sequential(self.s, model_table = 'Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path = self.data_dir + 'images.sashdat', task = 'load')

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'eee', 'replace': True},
                               path = path)
        solver = VanillaSolver(CyclicLR(self.s, 'eee', 4, 1.0, 0.001, 0.01))
        # solver = VanillaSolver(FixedLR(0.1))
        optimizer = Optimizer(algorithm = solver, log_level = 3, max_epochs = 4, mini_batch_size = 2)
        r = model1.fit(data = 'eee', inputs = '_image_', target = '_label_', optimizer = optimizer, n_threads=2)
        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertTrue(r.severity <= 1)

    def test_ReduceLROnPlateau(self):
        model1 = Sequential(self.s, model_table = 'Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2))

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path = caslibify(self.s, path = self.data_dir + 'images.sashdat', task = 'load')

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'eee', 'replace': True},
                               path = path)
        solver = VanillaSolver(ReduceLROnPlateau(self.s, 0.1, 0.1))
        # solver = VanillaSolver(FixedLR(0.1))
        optimizer = Optimizer(algorithm = solver, log_level = 2, max_epochs = 4, mini_batch_size = 10)
        r = model1.fit(data='eee', inputs='_image_', target='_label_', optimizer=optimizer)
        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertTrue(r.severity <= 1)


