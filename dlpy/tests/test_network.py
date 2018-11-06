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
from dlpy.network import Network
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

        cls.s = swat.CAS('dlgrd009', 13322)
        cls.server_type = tm.get_cas_host_type(cls.s)

    def test_option_type(self):
        input1 = InputLayer(n_channels = 1, width = 28, height = 28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        concat1 = Concat(src_layers = [conv1, conv2])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Network(conn = self.s, inputs = [input1], outputs = output1)
        model1.compile()
        model1.print_summary()

        model2 = Network(conn = self.s, inputs = input1, outputs = output1)
        model2.compile()
        model2.print_summary()

    def test_concat(self):
        input1 = InputLayer(n_channels = 1, width = 28, height = 28)
        input2 = InputLayer(n_channels = 3, width = 28, height = 28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat(src_layers = [conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Network(conn = self.s, inputs = [input1, input2], outputs = [output1, output2])
        model1.compile()
        model1.print_summary()

    def test_inputs(self):
        try:
            conv1 = Conv2d(2)
            output1 = OutputLayer(2)(conv1)
            model1 = Network(conn = self.s, inputs = [conv1], outputs = [output1])
            model1.compile()
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

    def test_outputs(self):
        try:
            input1 = InputLayer(n_channels = 1, width = 28, height = 28)
            conv1 = Conv2d(2)(input1)
            model1 = Network(conn = self.s, inputs = [input1], outputs = [conv1])
            model1.compile()
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

        try:
            input1 = InputLayer(n_channels = 1, width = 28, height = 28)
            conv1 = Conv2d(2)(input1)
            conv2 = BN()(input1)
            output1 = OutputLayer(2)(conv1)
            model1 = Network(conn = self.s, inputs = [input1], outputs = [conv2, output1])
            model1.compile()
        except DLPyError:
            pass
        except Exception as e:
            self.fail('Unexpected exception raised:', e)
        else:
            self.fail('ExpectedException not raised')

    def test_without_variable(self):
        input1 = InputLayer(n_channels = 1, width = 28, height = 28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        output1 = OutputLayer(n=2)(conv1)
        model1 = Network(conn = self.s, inputs = [input1], outputs = [output1])
        model1.compile()
        model1.print_summary()

    def test_multiple_inputs_outputs(self):
        input1 = InputLayer(n_channels = 1, width = 28, height = 28)
        input2 = InputLayer(n_channels = 3, width = 28, height = 28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat(src_layers = [conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Network(conn = self.s, inputs = [input1, input2], outputs = [output1, output2])
        model1.compile()
        model1.print_summary()

    def test_unet(self):
        inputs = InputLayer(3, 512, 512, scale = 1.0 / 255)
        conv1 = Conv2d(8, 3, act = 'relu')(inputs)
        conv1 = Conv2d(8, 3, act = 'relu')(conv1)
        pool1 = Pooling(2)(conv1)
        # 256
        conv2 = Conv2d(16, 3, act = 'relu')(pool1)
        conv2 = Conv2d(16, 3, act = 'relu')(conv2)
        pool2 = Pooling(2)(conv2)
        # 128
        conv3 = Conv2d(32, 3, act = 'relu')(pool2)
        conv3 = Conv2d(32, 3, act = 'relu')(conv3)
        pool3 = Pooling(2)(conv3)
        # 64
        conv4 = Conv2d(64, 3, act = 'relu')(pool3)  # 64

        tconv1 = Transconvo(32, 3, stride = 2, padding = 1, output_size = (128, 128, 32))(conv4)  # 128
        merge1 = Concat(src_layers = [conv3, tconv1])
        conv5 = Conv2d(32, 3, act = 'relu')(merge1)
        conv5 = Conv2d(32, 3, act = 'relu')(conv5)

        tconv2 = Transconvo(32, 3, stride = 2, padding = 1, output_size = (256, 256, 32))(conv5)  # 256
        merge2 = Concat(src_layers = [conv2, tconv2])
        conv6 = Conv2d(16, 3, act = 'relu')(merge2)
        conv6 = Conv2d(16, 3, act = 'relu')(conv6)

        tconv3 = Transconvo(32, stride = 2, padding = 1, output_size = (512, 512, 32))(conv6)  # 512
        merge3 = Concat(src_layers = [conv1, tconv3])
        conv7 = Conv2d(8, 3, act = 'relu')(merge3)
        conv7 = Conv2d(8, 3, act = 'relu')(conv7)

        conv8 = Conv2d(2, 3, act = 'relu')(conv7)

        seg1 = Segmentation()(conv8)
        model = Network(self.s, inputs = inputs, outputs = seg1)
        model.compile()
        model.print_summary()

    def test_network_transpose_conv(self):
        inputs = InputLayer(3, 128, 64, scale = 0.004, name = 'input1')
        tconv1 = Transconvo(32, height = 5, width = 3, stride = 2, padding_height = 2,
                            padding_width = 1, output_size = (256, 128, 32), name = 'trans1')(inputs)
        seg1 = Segmentation(name = 'seg1')(tconv1)
        model = Network(self.s, inputs = inputs, outputs = seg1)
        model.compile()
        model.print_summary()

    def test_given_name(self):
        inputs = InputLayer(3, 512, 512, scale = 1.0 / 255, name = 'input1')
        conv1 = Conv2d(8, 3, act = 'relu')(inputs)
        conv2 = Conv2d(8, 3, act = 'relu')(inputs)
        merge3 = Concat(src_layers = [conv1, conv2])
        conv3 = Conv2d(3, 3, act = 'relu')(merge3)
        seg1 = Segmentation(name = 'seg1')(conv3)
        model = Network(self.s, inputs = inputs, outputs = seg1)
        model.compile()
        self.assertTrue(inputs.name == 'input1')
        self.assertTrue(seg1.name == 'seg1')
        model.print_summary()

    def test_super_resolution(self):
        def conv_block(x, filters, size, stride = 1, mode = 'same', act = True):
            x = Conv2d(filters, size, size, act = 'identity', include_bias = False, stride = stride)(x)
            x = BN(act = 'relu' if act else 'identity')(x)
            return x

        def res_block(ip, nf = 64):
            x = conv_block(ip, nf, 3, 1)
            x = conv_block(x, nf, 3, 1, act = False)
            return Concat(src_layers = [x, ip])

        def deconv_block(x, filters, size, shape, stride = 2):
            x = Transconvo(filters, size, size, act = 'identity', padding=1, include_bias = False, stride = stride,
                           output_size = shape)(x)
            x = BN(act = 'relu')(x)
            return x

        inp = InputLayer(1, 32, 32, scale = 1.0 / 255, name = 'InputLayer_1')
        x = conv_block(inp, 64, 9, 1)
        for i in range(4): x = res_block(x)
        x = deconv_block(x, 64, 3, (64, 64, 64))
        x = deconv_block(x, 64, 3, (128, 128, 64))
        x = Conv2d(3, 9, 9, act = 'tanh')(x)
        seg = Segmentation(name = 'Segmentation_1', act = 'auto')(x)
        model = Network(self.s, inputs = inp, outputs = seg)
        model.compile()

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()

