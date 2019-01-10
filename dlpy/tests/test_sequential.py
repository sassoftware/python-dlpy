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

import os
import swat
import swat.utils.testing as tm
from dlpy.sequential import Sequential
from dlpy.layers import *
from dlpy.blocks import Bidirectional
from dlpy.utils import DLPyError


class TestSequential(tm.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.s = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.s)

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

    def test_1(self):
        with self.assertRaises(DLPyError):
            Sequential(self.s, layers='', model_table='table1')

    def test_11(self):
        with self.assertRaises(DLPyError):
            model1 = Sequential(self.s, model_table='table11')
            model1.add(Conv2d(8, 7))
            model1.compile()

    def test_2(self):
        layers = [InputLayer(), Dense(n=32), OutputLayer()]
        Sequential(self.s, layers=layers, model_table='table2')

    def test_22(self):
        layers = [InputLayer(3, 4), Conv2d(8, 7), BN(), Dense(n=32), OutputLayer()]
        Sequential(self.s, layers=layers, model_table='table22')

    def test_3(self):
        model1 = Sequential(self.s, model_table='table3')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.pop()

    def test_4(self):
        model1 = Sequential(self.s, model_table='table4')
        model1.add(Conv2d(8, 7))
        model1.add(InputLayer(3, 224, 224))
        model1.switch(0, 1)

    def test_5(self):
        with self.assertRaises(DLPyError):
            model1 = Sequential(self.s, model_table='table5')
            model1.compile()

    def test_6(self):
        model1 = Sequential(self.s, model_table='table6')
        model1.add(Bidirectional(n=10, n_blocks=3))
        model1.add(OutputLayer())

    def test_simple_cnn_seq1(self):
        model1 = Sequential(self.s, model_table='table7')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

    def test_simple_cnn_seq2(self):
        model1 = Sequential(self.s, model_table='table8')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))
        model1.print_summary()

    def test_new_bidirectional1(self):
        model = Sequential(self.s, model_table='new_table1')
        model.add(Bidirectional(n=10))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional2(self):
        model = Sequential(self.s, model_table='new_table2')
        model.add(Bidirectional(n=10, n_blocks=3))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional3(self):
        model = Sequential(self.s, model_table='new_table3')
        model.add(Bidirectional(n=[10, 20, 30], n_blocks=3))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional4(self):
        model = Sequential(self.s, model_table='new_table4')
        model.add(InputLayer())
        model.add(Recurrent(n=10, name='rec1'))
        model.add(Bidirectional(n=20, src_layers=['rec1']))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional5(self):
        model = Sequential(self.s, model_table='new_table5')
        model.add(InputLayer())
        model.add(Recurrent(n=10, name='rec1'))
        model.add(Bidirectional(n=20, src_layers=['rec1']))
        model.add(Recurrent(n=10))
        model.add(OutputLayer())
        model.print_summary()

    def test_new_bidirectional6(self):
        model = Sequential(self.s, model_table='new_table5')
        model.add(InputLayer())
        r1 = Recurrent(n=10, name='rec1')
        model.add(r1)
        model.add(Bidirectional(n=20, src_layers=[r1]))
        model.add(Recurrent(n=10))
        model.add(OutputLayer())
        model.print_summary()
