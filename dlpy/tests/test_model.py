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

import swat
import swat.utils.testing as tm
from dlpy.Sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, Pooling, Dense, OutputLayer

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()


class TestImageTable(tm.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS(HOST, PORT, USER, PASSWD, protocol=PROTOCOL)

        if type(self).server_type is None:
            # Set once per class and have every test use it. No need to change between tests.
            type(self).server_type = tm.get_cas_host_type(self.s)

        self.srcLib = tm.get_casout_lib(self.server_type)

        # Define the model
        model = Sequential(self.s, model_table='test_model')
        model.add(InputLayer(3, 224, 224, offsets=(0, 0, 0)))
        model.add(Conv2d(8, 7))
        model.add(Pooling(2))
        model.add(Conv2d(8, 7))
        model.add(Pooling(2))
        model.add(Dense(16))
        model.add(OutputLayer(act='softmax', n=2))

        self.model = model

    def tearDown(self):
        # tear down tests
        try:
            self.s.endsession()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()

    def test_model_type(self):
        self.assertTrue(isinstance(self.model, Sequential))
        self.assertEqual(self.model.model_table['name'], 'test_model')

    def test_model_info(self):
        out = self.model.get_model_info().ModelInfo
        column_list = ['Model Name', 'Model Type', 'Number of Layers',
                       'Number of Input Layers', 'Number of Output Layers',
                       'Number of Convolutional Layers', 'Number of Pooling Layers',
                       'Number of Fully Connected Layers']
        value_list = ['test_model', 'Convolutional Neural Network', '7', '1', '1', '2', '2', '1']

        for i, v in enumerate(column_list):
            self.assertEqual(out.loc[i].Descr, v)
        for i, v in enumerate(value_list):
            self.assertEqual(out.loc[i].Value.strip(), v)

    def test_layer_types(self):
        layer_type_list = [InputLayer,
                           Conv2d,
                           Pooling,
                           Conv2d,
                           Pooling,
                           Dense,
                           OutputLayer]
        for layer, layer_type in zip(self.model.layers, layer_type_list):
            self.assertTrue(isinstance(layer, layer_type))

    def test_layer_config(self):
        config_keys = [['offsets', 'dropout', 'nchannels', 'width', 'height', 'scale', 'type'],
                       ['act', 'nfilters', 'dropout', 'stride', 'width', 'height', 'type'],
                       ['pool', 'dropout', 'stride', 'width', 'height', 'type'],
                       ['act', 'nfilters', 'dropout', 'stride', 'width', 'height', 'type'],
                       ['pool', 'dropout', 'stride', 'width', 'height', 'type'],
                       ['act', 'dropout', 'n', 'type'],
                       ['act', 'n', 'type']]
        config_values = [[(0, 0, 0), 0, 3, 224, 224, 1, 'input'],
                         ['relu', 8, 0, 1, 7, 7, 'convo'],
                         ['max', 0, 2, 2, 2, 'pool'],
                         ['relu', 8, 0, 1, 7, 7, 'convo'],
                         ['max', 0, 2, 2, 2, 'pool'],
                         ['relu', 0, 16, 'fc'],
                         ['softmax', 2, 'output']]
        for layer, keys, values in zip(self.model.layers, config_keys, config_values):
            for key, value in zip(keys, values):
                self.assertEqual(layer.config[key], value)


if __name__ == '__main__':
    tm.runtests()
