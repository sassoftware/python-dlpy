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
from dlpy.model import Model
from dlpy.layers import *
from dlpy.utils import DLPyError


class TestNetwork(tm.TestCase):
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

    def test_option_type(self):
        input1 = InputLayer(n_channels = 1, width = 28, height = 28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        concat1 = Concat(src_layers = [conv1, conv2])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Model(conn = self.s, inputs = [input1], outputs = output1)
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 6314)

        model2 = Model(conn = self.s, inputs = input1, outputs = output1)
        model2.compile()
        model2.print_summary()
        model2.plot_network()
        self.assertTrue(model2.count_params() == 6314)

    def test_concat(self):
        input1 = InputLayer(n_channels=1, width=28, height=28)
        input2 = InputLayer(n_channels=3, width=28, height=28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat(src_layers=[conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Model(conn=self.s, inputs=[input1, input2], outputs=[output1, output2])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 12644)

    def test_inputs(self):
        conv1 = Conv2d(2)
        output1 = OutputLayer(2)(conv1)
        with self.assertRaises(DLPyError):
            Model(conn=self.s, inputs=[conv1], outputs=[output1])

    def test_outputs(self):
        with self.assertRaises(DLPyError):
            input1 = InputLayer(n_channels=1, width=28, height=28)
            conv1 = Conv2d(2)(input1)
            model1 = Model(conn=self.s, inputs=[input1], outputs=[conv1])
            model1.compile()

        with self.assertRaises(DLPyError):
            input1 = InputLayer(n_channels=1, width=28, height=28)
            conv1 = Conv2d(2)(input1)
            conv2 = BN()(input1)
            output1 = OutputLayer(2)(conv1)
            model1 = Model(conn=self.s, inputs=[input1], outputs=[conv2, output1])
            model1.compile()

    def test_without_variable(self):
        input1 = InputLayer(n_channels=1, width=28, height=28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        output1 = OutputLayer(n=2)(conv1)
        model1 = Model(conn=self.s, inputs=[input1], outputs=[output1])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 3196)

    def test_multiple_inputs_outputs(self):
        input1 = InputLayer(n_channels=1, width=28, height=28)
        input2 = InputLayer(n_channels=3, width=28, height=28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat(src_layers=[conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Model(conn=self.s, inputs=[input1, input2], outputs=[output1, output2])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 12682)

    def test_given_name(self):
        inputs = InputLayer(3, 512, 512, scale=1.0 / 255, name='input1')
        conv1 = Conv2d(8, 3, act='relu')(inputs)
        conv2 = Conv2d(8, 3, act='relu')(inputs)
        merge3 = Concat(src_layers=[conv1, conv2])
        conv3 = Conv2d(3, 3, act='relu')(merge3)
        output1 = OutputLayer(name='output1')(conv3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(inputs.name == 'input1')
        self.assertTrue(output1.name == 'output1')
        model.print_summary()

    def test_multi_src_layer_fc(self):
        inputs = InputLayer(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128)(inputs)
        fc2 = Dense(n=64)(fc1)
        fc3 = Dense(n=64)([fc1, fc2])
        output1 = OutputLayer(n=10, name='OutputLayer_1')(fc3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(model.count_params() == 117386)

    def test_single_src_layer_list_fc(self):
        inputs = InputLayer(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128)(inputs)
        fc3 = Dense(n=64)([fc1])
        output1 = OutputLayer(n=10, name='OutputLayer_1')(fc3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(model.count_params() == 109194)

    def test_non_list_src_layers(self):
        inputs = InputLayer(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128, src_layers=inputs)(inputs)
        fc2 = Dense(n=64)(fc1)
        output1 = OutputLayer(n=10, name='OutputLayer_1', src_layers=[fc1, fc2])
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 109834)

    def test_duplicated_src_layers_call(self):
        inputs = InputLayer(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128, src_layers=inputs)(inputs)
        fc2 = Dense(n=64)(fc1)
        output1 = OutputLayer(n=10, name='OutputLayer_1', src_layers=fc1)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(len(fc1.src_layers) == 1)

    def test_raise_conv_multi_src(self):
        with self.assertRaises(DLPyError):
            inputs = InputLayer(3, 512, 512, scale=1.0 / 255, name='input1')
            conv1 = Conv2d(8, 3, act='relu')(inputs)
            conv2 = Conv2d(8, 3, act='relu')(inputs)
            conv3 = Conv2d(8, 3)([conv1, conv2])
            output1 = OutputLayer(name='output1')(conv3)
            model = Model(self.s, inputs=inputs, outputs=output1)
            model.compile()

    def test_lack_inputs_outputs(self):
        with self.assertRaises(DLPyError):
            inputs = InputLayer(3, 512, 512, scale=1.0 / 255, name='input1')
            conv1 = Conv2d(8, 3, act='relu')(inputs)
            conv2 = Conv2d(8, 3, act='relu')(inputs)
            conv3 = Conv2d(8, 3)([conv1])
            output1 = OutputLayer(name='output1')(conv3)
            model = Model(self.s, inputs=inputs)
            model.compile()

        with self.assertRaises(DLPyError):
            inputs = InputLayer(3, 512, 512, scale=1.0 / 255, name='input1')
            conv1 = Conv2d(8, 3, act='relu')(inputs)
            conv2 = Conv2d(8, 3, act='relu')(inputs)
            conv3 = Conv2d(8, 3)([conv1])
            output1 = OutputLayer(name='output1')(conv3)
            model = Model(self.s, outputs=output1)
            model.compile()

    def test_sub_network(self):
        inputs1 = InputLayer(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10)
        dense2 = Dense(12)(dense1)
        dense3 = Dense(n=128)(dense2)
        model_dense = Model(self.s, inputs=dense1, outputs= dense3)

        node1 = model_dense(inputs1)
        concat1 = Dense(20, src_layers = [node1])
        output1 = OutputLayer(n=10, name='OutputLayer_1', src_layers=concat1)
        model = Model(self.s, inputs=[inputs1], outputs=output1)
        model.compile()
        model.print_summary()

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

