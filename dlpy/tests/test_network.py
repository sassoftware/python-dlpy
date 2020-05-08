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
from dlpy import Sequential
import unittest


class TestNetwork(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.s)

        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            self.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if self.data_dir_local.endswith(self.server_sep):
                self.data_dir_local = self.data_dir_local[:-1]
            self.data_dir_local += self.server_sep

    def test_network_option_type(self):
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

    def test_multiple_branch(self):
        from dlpy.sequential import Sequential
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(InputLayer(3, 48, 96, scale = 1 / 255, random_mutation = 'none'))
        model.add(Conv2d(64, 7, include_bias = True, act = 'relu'))
        model.add(Pooling(2))
        model.add(Conv2d(64, 3, include_bias = True, act = 'relu'))
        model.add(Pooling(2))
        model.add(Conv2d(64, 3, include_bias = True, act = 'relu'))
        model.add(Pooling(2))
        model.add(Conv2d(64, 3, include_bias = True, act = 'relu'))
        model.add(Pooling(2))
        model.add(Dense(16))
        model.add(OutputLayer(n = 1, act = 'sigmoid'))
        branch = model.to_functional_model(stop_layers = model.layers[-1])
        inp1 = Input(**branch.layers[0].config)  # tensor
        branch1 = branch(inp1)  # tensor
        inp2 = Input(**branch.layers[0].config)  # tensor
        branch2 = branch(inp2)  # tensor
        inp3 = Input(**branch.layers[0].config)  # tensor
        branch3 = branch(inp3)  # tensor
        triplet = OutputLayer(n = 1)(branch1 + branch2 + branch3)
        triplet_model = Model(self.s, inputs = [inp1, inp2, inp3], outputs = triplet)
        triplet_model.compile()
        triplet_model.print_summary()
        self.assertEqual(len(triplet_model.layers), 31)
        triplet_model.share_weights({'Convo.1': ['Convo.1_2', 'Convo.1_3']})
        triplet_model.compile()

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

    def test_stop_layers1(self):
        from dlpy.applications import MobileNetV1
        backbone = MobileNetV1(self.s, width = 1248, height = 1248)
        backbone_pure = backbone.to_functional_model(stop_layers = backbone.layers[-2])
        # expect last layer to be a bn layer right before global average pooling
        self.assertEqual(backbone_pure.output_layers[0].name, 'conv_pw_13_bn')

    def test_stop_layers2(self):
        from dlpy.applications import MobileNetV2
        backbone = MobileNetV2(self.s, width = 1248, height = 1248)
        backbone_pure = backbone.to_functional_model(stop_layers = backbone.layers[-14])
        # expect to get one outputs(block_15_depthwise) since stop layer is in a branch
        # output layer is not a valid output_layers since its dependencies cannot be traversed.
        self.assertEqual(len(backbone_pure.output_layers), 1)

    def test_astore_deploy_wrong_path(self):
        from dlpy.sequential import Sequential
        from dlpy.utils import caslibify
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(1, 17))
        model1.add(Pooling(14))
        model1.add(Dense(3))
        model1.add(OutputLayer(act='softmax', n=2))

        if self.data_dir is None:
            tm.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        r = model1.fit(data='eee', inputs='_image_', target='_label_', save_best_weights=True)

        self.assertTrue(r.severity == 0)

        with self.assertRaises(DLPyError):
            model1.deploy(path='/amran/komran', output_format='astore')

    def test_mix_cnn_rnn_network(self):
        from dlpy.applications import ResNet50_Caffe
        from dlpy import Sequential
        from dlpy.blocks import Bidirectional
        # the case is to test if CNN and RNN model can be connect using functional api
        # the model_type is expected to be RNN in 19w47.
        # CNN
        model = ResNet50_Caffe(self.s)
        cnn_head = model.to_functional_model(stop_layers = model.layers[-1])
        # RNN
        model_rnn = Sequential(conn = self.s, model_table = 'rnn')
        model_rnn.add(Bidirectional(n = 100, n_blocks = 2))
        model_rnn.add(OutputLayer('fixed'))

        f_rnn = model_rnn.to_functional_model()
        # connecting
        inp = Input(**cnn_head.layers[0].config)
        x = cnn_head(inp)
        y = f_rnn(x)
        cnn_rnn = Model(self.s, inp, y)
        cnn_rnn.compile()
        # check type
        self.assertTrue(cnn_rnn.model_type == 'RNN')
        self.assertTrue(cnn_rnn.layers[-1].name == 'fixed')
        self.assertTrue(x[0].shape == (1, 1, 2048))

        f_rnn = model_rnn.to_functional_model()
        # connecting
        inp = Input(**cnn_head.layers[0].config)
        x = cnn_head(inp)
        y = f_rnn(x)
        cnn_rnn = Model(self.s, inp, y)
        cnn_rnn.compile()
        # it should be fixed if I create f_rnn again.
        self.assertTrue(cnn_rnn.layers[-1].name == 'fixed')

        inp = Input(**cnn_head.layers[0].config)
        x = cnn_head(inp)
        y = f_rnn(x)
        cnn_rnn = Model(self.s, inp, y)
        cnn_rnn.compile()
        # it should be fixed if I create f_rnn again.
        self.assertTrue(cnn_rnn.layers[-1].name == 'fixed_2')

    def test_extract_segmentation1(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(InputLayer(3, 48, 96, scale = 1.0 / 255, random_mutation = 'none'))
        model.add(Dense(16))
        model.add(Segmentation(act='AUTO', target_scale=10))
        model_extracted = Model.from_table(self.s.CASTable(model.model_table['name']))
        model_extracted.compile()
        self.assertTrue(model_extracted.layers[-1].config['act'] == 'AUTO')
        self.assertTrue(model_extracted.layers[-1].config['target_scale'] == 10)

    def test_extract_embeddingloss1(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(InputLayer(3, 48, 96, scale = 1.0 / 255, random_mutation = 'none'))
        model.add(Dense(16))
        model.add(EmbeddingLoss(margin=10))
        model_extracted = Model.from_table(self.s.CASTable(model.model_table['name']))
        model_extracted.compile()
        self.assertTrue(model_extracted.layers[-1].config['margin'] == 10)

    def test_extract_clustering1(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(InputLayer(3, 48, 96, scale = 1.0 / 255, random_mutation = 'none'))
        model.add(Dense(16))
        model.add(Clustering(n_clusters=10))
        model_extracted = Model.from_table(self.s.CASTable(model.model_table['name']))
        model_extracted.compile()
        self.assertTrue(model_extracted.layers[-1].config['n_clusters'] == 10)

    def test_extract_clustering1(self):
        model = Sequential(self.s, model_table = 'Simple_CNN')
        model.add(InputLayer(3, 48, 96, scale = 1.0 / 255, random_mutation = 'none'))
        model.add(Split(16))
        model.add(Clustering(n_clusters=10))
        model_extracted = Model.from_table(self.s.CASTable(model.model_table['name']))
        model_extracted.compile()
        
    def test_multiple_stop_layers1(self):
        from dlpy.applications import ResNet50_Caffe
        resnet50 = ResNet50_Caffe(self.s, "res50")
        stop_layers = [resnet50.layers[x] for x in [-2, 4, -8] ]
        feature_extractor1 = resnet50.to_functional_model(stop_layers=stop_layers)
        
    def test_multiple_stop_layers2(self):
        from dlpy.applications import MobileNetV1
        resnet50 = MobileNetV1(self.s, "MobileNetV1")
        stop_layers = [resnet50.layers[x] for x in [-2, 4, -8] ]
        feature_extractor1 = resnet50.to_functional_model(stop_layers=stop_layers)

    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()


if __name__ == '__main__':
    unittest.main()
