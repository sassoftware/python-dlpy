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
import shutil
import swat
import swat.utils.testing as tm
from swat.cas.table import CASTable
from swat.cas.results import CASResults
from dlpy.model import Model, Optimizer
from dlpy.sequential import Sequential
from dlpy.tensorboard import TensorBoard
from dlpy.layers import (InputLayer, Conv2d, Conv1d, Pooling, Dense, OutputLayer,
                         Recurrent, Keypoints, BN, Res, Concat, Reshape, GlobalAveragePooling1D)
from dlpy.utils import caslibify, caslibify_context, file_exist_on_server, DLPyError
from dlpy.applications import Tiny_YoloV2
from dlpy.splitting import two_way_split
import unittest

class TestTensorBoard(unittest.TestCase):
    '''
    Please locate the images.sashdat file under the datasources to the DLPY_DATA_DIR.
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

    def test_tensorboard_init_log_dir(self):
        try:
            import tensorflow as tf
        except:
            unittest.TestCase.skipTest(self, "tensorflow not found in the libraries")

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

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)
        # Clean up for DNE
        shutil.rmtree(self.data_dir + '_TB', ignore_errors=True)

        # Test log_dir DNE
        self.assertRaises(OSError, lambda:TensorBoard(model1, self.data_dir + '_TB'))

        # Test existing log_dir
        os.mkdir(self.data_dir + '_TB')
        tensorboard = TensorBoard(model1, self.data_dir + '_TB')
        self.assertEqual(tensorboard.log_dir, self.data_dir + '_TB')

        # Clean up for next test
        shutil.rmtree(self.data_dir + '_TB', ignore_errors=True)
        shutil.rmtree(self.data_dir + '_TBSimple_CNN1', ignore_errors=True)

    def test_tensorboard_build_summary_writer(self):
        try:
            import tensorflow as tf
        except:
            unittest.TestCase.skipTest(self, "tensorflow not found in the libraries")

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

        if os.path.exists(self.data_dir + '_TB'):
            log_dir = self.data_dir + '_TB'
        else:
            os.mkdir(self.data_dir + '_TB')
            log_dir = self.data_dir + '_TB'

        # Test default scalars
        tensorboard = TensorBoard(model1, log_dir)
        writer = tensorboard.build_summary_writer()
        default_scalar_list = ['learning_rate', 'loss', 'error']
        for i in default_scalar_list:
            self.assertTrue(i in writer)
                        
        # Test with validation scalars
        tensorboard = TensorBoard(model1, log_dir, use_valid=True)
        valid_writer = tensorboard.build_summary_writer()
        valid_scalar_list = ['learning_rate', 'loss', 'error', 'valid_loss', 'valid_error']
        for i in valid_scalar_list:
            self.assertTrue(i in valid_writer)
                        
        # Clean up for next test
        shutil.rmtree(self.data_dir + '_TB', ignore_errors=True)
        shutil.rmtree(self.data_dir + '_TBSimple_CNN1', ignore_errors=True)

    def test_tensorboard_log_scalar(self):
        try:
            import tensorflow as tf
            import numpy as np
            from tensorflow.core.util import event_pb2
        except:
            unittest.TestCase.skipTest(self, "tensorflow and/or np not found in the libraries")
        
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

        if os.path.exists(self.data_dir + '_TB'):
            log_dir = self.data_dir + '_TB'
        else:
            os.mkdir(self.data_dir + '_TB')
            log_dir = self.data_dir + '_TB'

        # Generate test data for writing scalar values
        # np.random.seed(123)
        test_loss_values = list(range(20))
        # np.random.normal(size=10)
        tensorboard = TensorBoard(model1, log_dir)
        writers = tensorboard.build_summary_writer()
        
        # Write out test loss data as tfevents
        for i in range(10):
            tensorboard.log_scalar(writers['loss'], 'loss', test_loss_values[i], i)

        # Check event files for correct output data
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    print(t,test_loss_values[count])
                    self.assertAlmostEqual(t, test_loss_values[count], places=4)
                    count += 1

        # Continue logging
        for i in range(10,20):
            tensorboard.log_scalar(writers['loss'], 'loss', test_loss_values[i], i)

        # Check event files for correct output data after additional log
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, test_loss_values[count], places=4)
                    count += 1

        # Clean up for next test
        shutil.rmtree(self.data_dir + '_TB', ignore_errors=True)
        shutil.rmtree(self.data_dir + '_TBSimple_CNN1', ignore_errors=True)

    def test_tensorboard_response_cb(self):
        try:
            import tensorflow as tf
            import numpy as np
            from tensorflow.core.util import event_pb2
        except:
            unittest.TestCase.skipTest(self, "tensorflow and/or np not found in the libraries")
        
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")
        
        if os.path.exists(self.data_dir + '_TB'):
            log_dir = self.data_dir + '_TB'
        else:
            os.mkdir(self.data_dir + '_TB')
            log_dir = self.data_dir + '_TB'

        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        response = swat.cas.response.CASResponse(swat.cas.rest.response.REST_CASResponse({}),connection=self.s)
        userdata = None
        tensor_board = TensorBoard(model1, log_dir, use_valid=True)

        # Check initial values for userdata
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertTrue(isinstance(userdata, CASResults))
        self.assertEquals(len(userdata.message), 0)
        self.assertFalse(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 1)

        # Add a response message and check if passed to userdata
        response.messages.append('str1')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'str1')
        self.assertFalse(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 1)

        # Add another response message
        response.messages.pop()
        response.messages.append('str2')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'str2')
        self.assertFalse(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 1)

        # Check on Epoch changes at_scalar
        response.messages.pop()
        response.messages.append('Epoch')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'Epoch')
        self.assertTrue(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 1)

        # Check scalar values are logged and epoch increases
        response.messages.pop()
        response.messages.append('NOTE:          1            2       3        4          5              6           7')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'NOTE:          1            2       3        4          5              6           7')
        self.assertTrue(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 2)

        # Check for correct scalar values
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/learning_rate/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/learning_rate/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 2.0, places=4)
                    count += 1
            self.assertEquals(count, 1)

        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 3.0, places=4)
                    count += 1
            self.assertEquals(count, 1)
        
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/error/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/error/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 4.0, places=4)
                    count += 1
            self.assertEquals(count, 1)

        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/valid_loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/valid_loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 5.0, places=4)
                    count += 1
            self.assertEquals(count, 1)

        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/valid_error/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/valid_error/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 6.0, places=4)
                    count += 1
            self.assertEquals(count, 1)

        # Check on Batch
        response.messages.pop()
        response.messages.append('Batch')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'Batch')
        self.assertFalse(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 2)

        # Check on Epoch changes at_scalar
        response.messages.pop()
        response.messages.append('Epoch')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'Epoch')
        self.assertTrue(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 2)

        # Check scalar values are logged and epoch increases
        response.messages.pop()
        response.messages.append('NOTE:          1            2       3        4          5              6           7')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'NOTE:          1            2       3        4          5              6           7')
        self.assertTrue(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 3)

        # Check for correct scalar values
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/learning_rate/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/learning_rate/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 2.0, places=4)
                    count += 1
            self.assertEquals(count, 2)

        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 3.0, places=4)
                    count += 1
            self.assertEquals(count, 2)
        
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/error/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/error/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 4.0, places=4)
                    count += 1
            self.assertEquals(count, 2)
        
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/valid_loss/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/valid_loss/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 5.0, places=4)
                    count += 1
            self.assertEquals(count, 2)
        
        count = 0
        tfevent_file = os.listdir(log_dir +'Simple_CNN1' + '/valid_error/')
        for file in tfevent_file:
            serialized_examples = tf.data.TFRecordDataset(log_dir +'Simple_CNN1' + '/valid_error/' + file)
            for serialized_example in serialized_examples:
                event = event_pb2.Event.FromString(serialized_example.numpy())
                for value in event.summary.value:
                    t = tf.make_ndarray(value.tensor)
                    self.assertAlmostEqual(t, 6.0, places=4)
                    count += 1
            self.assertEquals(count, 2)

        # On optimization changes at_scalar
        response.messages.pop()
        response.messages.append('optimization')
        userdata = tensor_board.tensorboard_response_cb(response, self.s, userdata)
        self.assertEquals(len(userdata.message), 1)
        self.assertEquals(userdata.message[0], 'optimization')
        self.assertFalse(userdata.at_scaler)
        self.assertEquals(len(userdata.writer_dict), 5)
        self.assertEquals(userdata.epoch_count, 3)

        # Clean up for next test
        shutil.rmtree(self.data_dir + '_TB', ignore_errors=True)
        shutil.rmtree(self.data_dir + '_TBSimple_CNN1', ignore_errors=True)
        
    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
