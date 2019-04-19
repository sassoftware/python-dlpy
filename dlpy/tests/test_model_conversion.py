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
#import onnx
import swat
import swat.utils.testing as tm
from swat.cas.table import CASTable
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from dlpy.model import Model
from shutil import copyfile
import unittest


class TestModelConversion(unittest.TestCase):
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

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            cls.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if cls.data_dir_local.endswith(cls.server_sep):
                cls.data_dir_local = cls.data_dir_local[:-1]
            cls.data_dir_local += cls.server_sep

    # instantiate a Keras LeNet model and translate to DLPy/Viya model
    # NOTE: cannot attach weights unless both client and server share
    #       the same file system
    def test_model_conversion1(self):

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isfile(os.path.join(self.data_dir_local,'lenet.h5'))):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables or lenet.h5 file is missing")
            
        model = Sequential()
        model.add(Conv2D(20, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28,28,1), padding="same"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        
        model.load_weights(os.path.join(self.data_dir_local,'lenet.h5'))
        model.summary()
        
        model_name = 'lenet'
        model1 = Model.from_keras_model(conn=self.s, keras_model=model, output_model_table=model_name,
                                    include_weights=True, scale=1.0/255.0,
                                    input_weights_file=os.path.join(self.data_dir_local,'lenet.h5'))  
                                
        if os.path.isdir(self.data_dir):
            try:
                copyfile(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5'),os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'))
                copy_success = True
            except:
                print('Unable to copy weights file, skipping test of attaching weights')
                copy_success = False
                
            if copy_success:
                model1.load_weights(path=os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'),
                                    labels=False)
                os.remove(os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'))
        
        model1.print_summary()
        
        if os.path.isfile(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5')):
            os.remove(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5'))

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
