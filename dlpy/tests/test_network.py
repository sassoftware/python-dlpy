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

    def test_option_type(self):
        input1 = Input(n_channels = 1, width = 28, height = 28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        concat1 = Concat()([conv1, conv2])
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
        input1 = Input(n_channels=1, width=28, height=28)
        input2 = Input(n_channels=3, width=28, height=28)
        conv1 = Conv2d(2)(input1)
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat()([conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Model(conn=self.s, inputs=[input1, input2], outputs=[output1, output2])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 12644)

    def test_inputs(self):
        conv1 = Conv2d(2)
        with self.assertRaises(ValueError):
            output1 = OutputLayer(2)(conv1)
            Model(conn=self.s, inputs=[conv1], outputs=[output1])

    def test_outputs(self):
        with self.assertRaises(DLPyError):
            input1 = Input(n_channels=1, width=28, height=28)
            conv1 = Conv2d(2)(input1)
            model1 = Model(conn=self.s, inputs=[input1], outputs=[conv1])
            model1.compile()

        with self.assertRaises(DLPyError):
            input1 = Input(n_channels=1, width=28, height=28)
            conv1 = Conv2d(2)(input1)
            conv2 = BN()(input1)
            output1 = OutputLayer(2)(conv1)
            model1 = Model(conn=self.s, inputs=[input1], outputs=[conv2, output1])
            model1.compile()

    def test_without_variable(self):
        input1 = Input(n_channels=1, width=28, height=28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        output1 = OutputLayer(n=2)(conv1)
        model1 = Model(conn=self.s, inputs=[input1], outputs=[output1])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 3196)

    def test_multiple_inputs_outputs(self):
        input1 = Input(n_channels=1, width=28, height=28)
        input2 = Input(n_channels=3, width=28, height=28)
        conv1 = Conv2d(2)(Conv2d(2)(input1))
        conv2 = Conv2d(2)(input1)
        conv3 = Conv2d(2)(input2)
        output2 = OutputLayer(n=2)(conv3)
        concat1 = Concat()([conv1, conv2, conv3])
        output1 = OutputLayer(n=2)(concat1)
        model1 = Model(conn=self.s, inputs=[input1, input2], outputs=[output1, output2])
        model1.compile()
        model1.print_summary()
        self.assertTrue(model1.count_params() == 12682)

    def test_given_name(self):
        inputs = Input(3, 512, 512, scale=1.0 / 255, name='input1')
        conv1 = Conv2d(8, 3, act='relu')(inputs)
        conv2 = Conv2d(8, 3, act='relu')(inputs)
        merge3 = Concat()([conv1, conv2])
        conv3 = Conv2d(3, 3, act='relu')(merge3)
        output1 = OutputLayer(name='output1')(conv3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(inputs._op.name == 'input1')
        self.assertTrue(output1._op.name == 'output1')
        model.print_summary()

    def test_multi_src_layer_fc(self):
        inputs = Input(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128, name = 'dense1')(inputs)
        fc2 = Dense(n=64, name = 'dense2')(fc1)
        fc3 = Dense(n=64, name = 'dense3')([fc1, fc2])
        output1 = OutputLayer(n=10, name='OutputLayer_1')(fc3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(model.count_params() == 117386)

    def test_single_src_layer_list_fc(self):
        inputs = Input(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128)(inputs)
        fc3 = Dense(n=64)([fc1])
        output1 = OutputLayer(n=10, name='OutputLayer_1')(fc3)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        self.assertTrue(model.count_params() == 109194)

    def test_non_list_src_layers(self):
        inputs = Input(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128)(inputs)
        fc2 = Dense(n=64)(fc1)
        output1 = OutputLayer(n=10, name='OutputLayer_1')([fc1, fc2])
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 109834)

    def test_duplicated_src_layers_call(self):
        inputs = Input(1, 28, 28, scale=1.0 / 255, name='InputLayer_1')
        fc1 = Dense(n=128)(inputs)
        fc2 = Dense(n=64)(fc1)
        output1 = OutputLayer(n=10, name='OutputLayer_1')(fc1)
        model = Model(self.s, inputs=inputs, outputs=output1)
        model.compile()
        # self.assertTrue(len(fc1.src_layers) == 1)

    def test_raise_conv_multi_src(self):
        with self.assertRaises(DLPyError):
            inputs = Input(3, 512, 512, scale=1.0 / 255, name='input1')
            conv1 = Conv2d(8, 3, act='relu')(inputs)
            conv2 = Conv2d(8, 3, act='relu')(inputs)
            conv3 = Conv2d(8, 3)([conv1, conv2])
            output1 = OutputLayer(name='output1')(conv3)
            model = Model(self.s, inputs=inputs, outputs=output1)
            model.compile()

    # def test_lack_inputs_outputs(self):
    #     with self.assertRaises(DLPyError):
    #         inputs = Input(3, 512, 512, scale=1.0 / 255, name='input1')
    #         conv1 = Conv2d(8, 3, act='relu')(inputs)
    #         conv2 = Conv2d(8, 3, act='relu')(inputs)
    #         conv3 = Conv2d(8, 3)([conv1])
    #         output1 = OutputLayer(name='output1')(conv3)
    #         model = Model(self.s, inputs=inputs)
    #         model.compile()
    #
    #     with self.assertRaises(DLPyError):
    #         inputs = Input(3, 512, 512, scale=1.0 / 255, name='input1')
    #         conv1 = Conv2d(8, 3, act='relu')(inputs)
    #         conv2 = Conv2d(8, 3, act='relu')(inputs)
    #         conv3 = Conv2d(8, 3)([conv1])
    #         output1 = OutputLayer(name='output1')(conv3)
    #         model = Model(self.s, outputs=output1)
    #         model.compile()

    def test_submodel_as_input_network(self):
        inputs1 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10, name='dense1')(inputs1)
        dense2 = Dense(12, name='dense2')(dense1)
        dense3 = Dense(n=128, name='dense3')(dense2)
        model_dense = Model(self.s, inputs=inputs1, outputs= dense3)

        inputs2 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_2')
        out_submodel = model_dense(inputs2)

        concat1 = Dense(20, 'concat1')(out_submodel)
        output1 = OutputLayer(n=10, name='OutputLayer_1')(concat1)
        model = Model(self.s, inputs=[inputs2], outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 32516)

    def test_submodel_as_inputs_network(self):
        inputs1 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10, name = 'd1')(inputs1)
        dense2 = Dense(12, name = 'd2')(dense1)
        dense3 = Dense(n=128, name = 'd3')(dense2)
        model_dense = Model(self.s, inputs=inputs1, outputs= dense3)

        inputs2 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_2')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense4 = Dense(10, name = 'd4')(inputs2)
        dense5 = Dense(12, name = 'd5')(dense4)
        dense6 = Dense(n = 128, name = 'd6')(dense5)
        model_dense2 = Model(self.s, inputs = inputs2, outputs = dense6)

        inputs3 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_3')
        inputs4 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_4')
        out3 = model_dense(inputs3)
        out4 = model_dense2(inputs4)

        out5 = [out4[0], out3[0]]

        concat1 = Dense(20, name = 'd7')(out5)
        output1 = OutputLayer(n=10, name='OutputLayer_1')(concat1)
        model = Model(self.s, inputs=[inputs3, inputs4], outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 62262)

    def test_submodel_in_middle_network(self):
        inputs1 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        inputs2 = Input(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10, name = 'd1')(inputs2)
        dense2 = Dense(12, name = 'd2')(dense1)
        dense3 = Dense(n=128, name = 'd3')(dense2)
        model_dense = Model(self.s, inputs=inputs2, outputs= dense3)

        fore = model_dense(inputs1)

        concat1 = Dense(20, name = 'd7')(fore)
        output1 = OutputLayer(n = 10, name = 'OutputLayer_1')(concat1)

        model = Model(self.s, inputs=[inputs1], outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 32516)

    def test_submodel_multiple_inputs_network(self):
        inputs1 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10, name = 'd1')(inputs1)
        dense2 = Dense(12, name = 'd2')(dense1)
        dense3 = Dense(n=128, name = 'd3')(dense2)
        # model_dense = Model(self.s, inputs=inputs1, outputs= dense3)

        inputs2 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_2')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense4 = Dense(10, name = 'd4')(inputs2)
        dense5 = Dense(12, name = 'd5')(dense4)
        dense6 = Dense(n = 128, name = 'd6')(dense5)
        dense7 = Dense(n =190, name = 'cat1')([dense3, dense6])
        model_dense = Model(self.s, inputs = [inputs1, inputs2], outputs = [dense7])

        input3 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_3')
        input4 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_4')

        out = model_dense([input3, input4])

        dense7 = Dense(20, name = 'd7')(out)
        output1 = OutputLayer(n=10, name='OutputLayer_1')(dense7)
        model = Model(self.s, inputs=[input3, input4], outputs=output1)
        model.compile()
        model.print_summary()
        self.assertTrue(model.count_params() == 87822)

    def test_submodel_multiple_inputs_in_middle_network(self):
        inputs3 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_3')
        inputs4 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_4')

        inputs1 = Input(1, 53, 53, scale=1.0 / 255, name='InputLayer_1')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense1 = Dense(10, name = 'd1')(inputs1)
        dense2 = Dense(12, name = 'd2')(dense1)
        dense3 = Dense(n=128, name = 'd3')(dense2)
        # model_dense = Model(self.s, inputs=inputs1, outputs= dense3)

        inputs2 = Input(1, 53, 53, scale = 1.0 / 255, name = 'InputLayer_2')
        # inputs2 = InputLayer(1, 28, 28, scale = 1.0 / 255, name = 'InputLayer_2')
        dense4 = Dense(10, name = 'd4')(inputs2)
        dense5 = Dense(12, name = 'd5')(dense4)
        dense6 = Dense(n = 128, name = 'd6')(dense5)
        dense7 = Dense(n =190, name = 'cat1')([dense3, dense6])

        model_dense = Model(self.s, inputs = [inputs1, inputs2], outputs = [dense7])
        sub_model = model_dense([inputs3, inputs4])

        dense7 = Dense(20, name = 'd7')(sub_model)
        output1 = OutputLayer(n=10, name='OutputLayer_1')(dense7)
        model = Model(self.s, inputs=[inputs3, inputs4], outputs=output1)
        model.compile()
        model.print_summary()

    def test_resnet(self):
        def conv_block(x, filters, size, stride = 1, mode = 'same', act = True):
            x = Conv2d(filters, size, size, act = 'identity', include_bias = False, stride = stride)(x)
            x = BN(act = 'relu' if act else 'identity')(x)
            return x

        def res_block(ip, nf = 64):
            x = conv_block(ip, nf, 3, 1)
            x = conv_block(x, nf, 3, 1, act = False)
            return Res()([x, ip])

        inp = Input(1, 32, 32, scale = 1.0 / 255, name = 'InputLayer_1')
        x = conv_block(inp, 64, 9, 1)
        for i in range(1): x = res_block(x)
        x = Conv2d(1, 9, 9, act = 'tanh')(x)
        output = OutputLayer(n = 100)(x)
        model = Model(self.s, inputs = inp, outputs = output)
        model.compile()

    def test_sequential_conversion(self):
        from dlpy.sequential import Sequential
        model1 = Sequential(self.s)
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2))

        func_model = model1.to_functional_model()
        func_model.compile()
        func_model.print_summary()

    def test_stop_layers(self):
        from dlpy.sequential import Sequential
        model1 = Sequential(self.s)
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act = 'softmax', n = 2))

        inputlayer = model1.layers[0]
        inp = Input(**inputlayer.config)
        func_model = model1.to_functional_model(stop_layers = [model1.layers[-1]])
        x = func_model(inp)
        out = Keypoints(n=10)(x)
        func_model_keypoints = Model(self.s, inp, out)
        func_model_keypoints.compile()
        func_model_keypoints.print_summary()

    def test_from_model(self):
        from dlpy.applications import ResNet18_Caffe, VGG11
        vgg11 = VGG11(self.s)
        backbone1 = vgg11.to_functional_model(vgg11.layers[-2])
        self.assertEqual(backbone1.layers[-1].__class__.__name__, 'Dense')
        model_resnet18 = ResNet18_Caffe(self.s, n_classes = 6, random_crop = 'none', width = 400, height = 400)
        backbone2 = model_resnet18.to_functional_model(model_resnet18.layers[-3])
        self.assertEqual(backbone2.layers[-1].__class__.__name__, 'BN')

    def test_from_model(self):
        from dlpy.applications import ResNet18_Caffe, VGG11
        vgg11 = VGG11(self.s)
        backbone1 = vgg11.to_functional_model(vgg11.layers[-2])
        self.assertEqual(backbone1.layers[-1].__class__.__name__, 'Dense')
        model_resnet18 = ResNet18_Caffe(self.s, n_classes = 6, random_crop = 'none', width = 400, height = 400)
        backbone2 = model_resnet18.to_functional_model(model_resnet18.layers[-3])
        self.assertEqual(backbone2.layers[-1].__class__.__name__, 'BN')

    def test_residual_output_shape0(self):
        inLayer = Input(n_channels = 3, width = 32, height = 128,
                        name = 'input1', random_mutation = 'random', random_flip = 'HV')
        conv1 = Conv2d(32, 3, 3, name = 'conv1', act = 'relu', init = 'msra')([inLayer])
        conv2 = Conv2d(32, 5, 5, name = 'conv2', act = 'relu', init = 'msra')([inLayer])
        fc1 = Dense(3, name = 'fc1')([conv1])
        fc2 = Dense(3, name = 'fc2')([conv2])
        res = Res(name = 'res1')([fc1, fc2])
        outLayer = OutputLayer(n = 3, name = 'output')([res])
        model = Model(self.s, inLayer, outLayer)
        model.compile()
        self.assertEqual(model.summary['Output Size'].values[-2], 3)
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

