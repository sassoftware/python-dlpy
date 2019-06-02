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
import sys
import swat
import swat.utils.testing as tm
from dlpy.model import Model
from dlpy.model import DataSpec, DataSpecNumNomOpts
from dlpy.utils import input_table_check
from shutil import copyfile
import unittest
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# helper function: define RNN model with two recurrent layers using Keras
def define_keras_rnn_model(layer_type, bidirectional, rnn_size, feature_dim, output_dim):

    import keras
    from keras import backend as K
    from keras.models import Model as KerasModel
    from keras.models import Sequential
    from keras.layers import LSTM, Input, Lambda, Bidirectional, CuDNNLSTM, Dropout, TimeDistributed, Dense, SimpleRNN, GRU
    from keras.layers import CuDNNLSTM, CuDNNGRU
    from keras.activations import relu
    from keras.utils import multi_gpu_model
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

    if layer_type not in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
        return None
        
    # define CTC function for Keras implementation
    def ctc_lambda_func(args):
        y_pred, labels, input_seq_len, label_seq_len = args
        # the 2 is critical here since the first couple outputs of the RNN
        # tend to be garbage:
        y_pred = y_pred[:, 2:, :]
        ret = K.ctc_batch_cost(labels, y_pred, input_seq_len-2, label_seq_len)
        return ret                        

    # start building the Keras model
    input_data = Input(name='the_input', shape=(None, feature_dim))

    if layer_type == 'simplernn':
        if bidirectional:
            out_layer1 = Bidirectional(SimpleRNN(rnn_size, 
                                        activation='tanh', 
                                        use_bias=True, 
                                        kernel_initializer='glorot_uniform', 
                                        recurrent_initializer='orthogonal', 
                                        bias_initializer='RandomNormal',
                                        dropout=0.0, 
                                        recurrent_dropout=0.0, 
                                        return_sequences=True, 
                                        return_state=False, 
                                        stateful=False, 
                                        unroll=False,
                                        name='birnn1'), merge_mode='concat')(input_data)
            out_layer2 = Bidirectional(SimpleRNN(rnn_size, 
                                        activation='tanh', 
                                        use_bias=True, 
                                        kernel_initializer='glorot_uniform', 
                                        recurrent_initializer='orthogonal', 
                                        bias_initializer='RandomNormal', 
                                        dropout=0.0, 
                                        recurrent_dropout=0.0, 
                                        return_sequences=True, 
                                        return_state=False, 
                                        stateful=False, 
                                        unroll=False,
                                        name='birnn2'), merge_mode='concat')(out_layer1)
        else:
            out_layer1 = SimpleRNN(rnn_size, 
                             activation='tanh', 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             recurrent_initializer='orthogonal', 
                             bias_initializer='RandomNormal', 
                             dropout=0.0, 
                             recurrent_dropout=0.0, 
                             return_sequences=True, 
                             return_state=False, 
                             go_backwards=False, 
                             stateful=False, 
                             unroll=False,
                             name='birnn1')(input_data)
            out_layer2 = SimpleRNN(rnn_size, 
                             activation='tanh', 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             recurrent_initializer='orthogonal', 
                             bias_initializer='RandomNormal', 
                             dropout=0.0, 
                             recurrent_dropout=0.0, 
                             return_sequences=True, 
                             return_state=False, 
                             go_backwards=False, 
                             stateful=False, 
                             unroll=False,
                             name='birnn2')(out_layer1)
    elif layer_type == 'lstm':
        if bidirectional:
            out_layer1 = Bidirectional(LSTM(rnn_size, 
                                       return_sequences=True,
                                       kernel_initializer='he_normal', 
                                       bias_initializer='RandomNormal',
                                       unit_forget_bias=False,
                                       activation='tanh',
                                       recurrent_activation='sigmoid',
                                       name='birnn1'), merge_mode='concat')(input_data)
            out_layer2 = Bidirectional(LSTM(rnn_size, 
                                       return_sequences=True,
                                       kernel_initializer='he_normal', 
                                       bias_initializer='RandomNormal',
                                       unit_forget_bias=False,
                                       activation='tanh',
                                       recurrent_activation='sigmoid',
                                       name='birnn2'), merge_mode='concat')(out_layer1)
        else:
            out_layer1 = LSTM(rnn_size, 
                              return_sequences=True,
                              kernel_initializer='he_normal',  
                              bias_initializer='RandomNormal',
                              unit_forget_bias=False,
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              name='birnn1')(input_data)
            out_layer2 = LSTM(rnn_size, 
                              return_sequences=True,
                              kernel_initializer='he_normal',  
                              bias_initializer='RandomNormal',
                              unit_forget_bias=False,
                              activation='tanh',
                              recurrent_activation='sigmoid',
                              name='birnn2')(out_layer1)
    elif layer_type == 'cudnnlstm':
        if bidirectional:
            out_layer1 = Bidirectional(CuDNNLSTM(rnn_size, 
                                               kernel_initializer='glorot_uniform', 
                                               recurrent_initializer='orthogonal', 
                                               bias_initializer='RandomNormal', 
                                               unit_forget_bias=True, 
                                               return_sequences=True, 
                                               return_state=False, 
                                               stateful=False,
                                               name='birnn1'), merge_mode='concat')(input_data)
            out_layer2 = Bidirectional(CuDNNLSTM(rnn_size, 
                                               kernel_initializer='glorot_uniform', 
                                               recurrent_initializer='orthogonal', 
                                               bias_initializer='RandomNormal', 
                                               unit_forget_bias=True, 
                                               return_sequences=True, 
                                               return_state=False, 
                                               stateful=False,
                                               name='birnn2'), merge_mode='concat')(out_layer1)     
        else:
            out_layer1 = CuDNNLSTM(rnn_size, 
                                   kernel_initializer='glorot_uniform', 
                                   recurrent_initializer='orthogonal', 
                                   bias_initializer='RandomNormal', 
                                   unit_forget_bias=True, 
                                   return_sequences=True, 
                                   return_state=False, 
                                   stateful=False,
                                   name='birnn1')(input_data)
            out_layer2 = CuDNNLSTM(rnn_size, 
                                   kernel_initializer='glorot_uniform', 
                                   recurrent_initializer='orthogonal', 
                                   bias_initializer='RandomNormal', 
                                   unit_forget_bias=True, 
                                   return_sequences=True, 
                                   return_state=False, 
                                   stateful=False,
                                   name='birnn2')(out_layer1)
    elif layer_type == 'gru':
        if bidirectional:
            out_layer1 = Bidirectional(GRU(rnn_size, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         use_bias=True, 
                                         kernel_initializer='glorot_uniform', 
                                         recurrent_initializer='orthogonal', 
                                         bias_initializer='RandomNormal', 
                                         dropout=0.0, 
                                         recurrent_dropout=0.0, 
                                         implementation=1, 
                                         return_sequences=True, 
                                         return_state=False, 
                                         go_backwards=False, 
                                         stateful=False, 
                                         unroll=False, 
                                         reset_after=False,    # set to True for CUDNN implementation
                                         name='birnn1'), merge_mode='concat')(input_data)
            out_layer2 = Bidirectional(GRU(rnn_size, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         use_bias=True, 
                                         kernel_initializer='glorot_uniform', 
                                         recurrent_initializer='orthogonal', 
                                         bias_initializer='RandomNormal', 
                                         dropout=0.0, 
                                         recurrent_dropout=0.0, 
                                         implementation=1, 
                                         return_sequences=True, 
                                         return_state=False, 
                                         go_backwards=False, 
                                         stateful=False, 
                                         unroll=False, 
                                         reset_after=False,    # set to True for CUDNN implementation
                                         name='birnn2'), merge_mode='concat')(out_layer1)
        else:
            out_layer1 = GRU(rnn_size, 
                             activation='tanh', 
                             recurrent_activation='sigmoid', 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             recurrent_initializer='orthogonal', 
                             bias_initializer='RandomNormal', 
                             dropout=0.0, 
                             recurrent_dropout=0.0, 
                             implementation=1, 
                             return_sequences=True, 
                             return_state=False, 
                             go_backwards=False, 
                             stateful=False, 
                             unroll=False, 
                             reset_after=False,    # set to True for CUDNN implementation
                             name='birnn1')(input_data)
            out_layer2 = GRU(rnn_size, 
                             activation='tanh', 
                             recurrent_activation='sigmoid', 
                             use_bias=True, 
                             kernel_initializer='glorot_uniform', 
                             recurrent_initializer='orthogonal', 
                             bias_initializer='RandomNormal', 
                             dropout=0.0, 
                             recurrent_dropout=0.0, 
                             implementation=1, 
                             return_sequences=True, 
                             return_state=False, 
                             go_backwards=False, 
                             stateful=False, 
                             unroll=False, 
                             reset_after=False,    # set to True for CUDNN implementation
                             name='birnn2')(out_layer1)
    elif layer_type == 'cudnngru':
        if bidirectional:        
            out_layer1 = Bidirectional(CuDNNGRU(rnn_size, 
                                                kernel_initializer='glorot_uniform', 
                                                recurrent_initializer='orthogonal', 
                                                bias_initializer='zeros', 
                                                return_sequences=True, 
                                                return_state=False, 
                                                stateful=False,
                                                name='birnn1'), merge_mode='concat')(input_data)
            out_layer2 = Bidirectional(CuDNNGRU(rnn_size, 
                                                kernel_initializer='glorot_uniform', 
                                                recurrent_initializer='orthogonal', 
                                                bias_initializer='zeros', 
                                                return_sequences=True, 
                                                return_state=False, 
                                                stateful=False,
                                                name='birnn2'), merge_mode='concat')(out_layer1)
        else:
            out_layer1 = CuDNNGRU(rnn_size, 
                                kernel_initializer='glorot_uniform', 
                                recurrent_initializer='orthogonal', 
                                bias_initializer='zeros', 
                                return_sequences=True, 
                                return_state=False, 
                                stateful=False,
                                name='birnn1')(input_data)
            out_layer2 = CuDNNGRU(rnn_size, 
                                kernel_initializer='glorot_uniform', 
                                recurrent_initializer='orthogonal', 
                                bias_initializer='zeros', 
                                return_sequences=True, 
                                return_state=False, 
                                stateful=False,
                                name='birnn2')(out_layer1)


    y_pred = TimeDistributed(Dense(output_dim, 
                                   name="y_pred", 
                                   kernel_initializer='he_normal', 
                                   bias_initializer='RandomNormal', # zeros 
                                   activation="softmax"), 
                                   name="out")(out_layer2)

    # Input of labels and other CTC requirements
    labels = Input(name='the_labels', shape=[None,], dtype='int32')
    input_length = Input(name='input_length', shape=[1], dtype='int32')
    label_length = Input(name='label_length', shape=[1], dtype='int32')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred,
                                                                       labels,
                                                                       input_length,
                                                                       label_length])

    model = KerasModel(inputs=[input_data, labels, input_length, label_length], outputs=[loss_out])

    adam = keras.optimizers.Adam(lr=1e-3, clipvalue=10000, clipnorm=5., epsilon=1e-8)
    model.compile(optimizer=adam, loss={'ctc': lambda y_true, y_pred: y_pred})
    
    return model

class TestModelConversion(unittest.TestCase):
    '''
    Please locate the images.sashdat file under the datasources to the DLPY_DATA_DIR.
    '''
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None
    
    try:
        stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        import keras
        keras_installed = True
        sys.stderr = stderr
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    except:
        keras_installed = False

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
            
        # deepLearn action set must be loaded
        cls.s.loadactionset(actionSet='deeplearn', _messagelevel='error')
            

    def test_model_conversion1(self):
        '''
        Import CNN image classification model
          - instantiate a Keras LeNet model and translate to DLPy/Viya model
        NOTE: cannot attach weights unless both client and server share
              the same file system
        COVERAGE: from_keras_model(), load_weights() in network.py
                  keras_to_sas() in sas_keras_parse.py
                  write_keras_hdf5_from_file() in write_keras_model_parm.py
                  all functions in model_conversion_utils.py
                  CNN-related function in write_sas_code.py
        '''

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isfile(os.path.join(self.data_dir_local,'lenet.h5'))):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables or lenet.h5 file is missing")

        if self.keras_installed:
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
        else:
            unittest.TestCase.skipTest(self, "keras is not installed")

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
        model1,use_gpu = Model.from_keras_model(conn=self.s, keras_model=model, output_model_table=model_name,
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
                                    labels=False, use_gpu=use_gpu)
                os.remove(os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'))
        
        model1.print_summary()
                
        if os.path.isfile(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5')):
            os.remove(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5'))

        # clean up model table
        model_tbl_opts = input_table_check(model_name)
        self.s.table.droptable(quiet=True, **model_tbl_opts)
        
        # clean up models
        del model
        del model1

    def test_model_conversion2(self):
        '''
        Import RNN sequence to sequence models
          - instantiate Keras RNN models and translate to DLPy/Viya models
        NOTE: cannot attach weights unless both client and server share
              the same file system
        COVERAGE: from_keras_model(), load_weights() in network.py
                  keras_to_sas() in sas_keras_parse.py
                  write_keras_hdf5() in write_keras_model_parm.py
                  all functions in model_conversion_utils.py
                  CNN-related function in write_sas_code.py
        '''

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isfile(os.path.join(self.data_dir_local,'lenet.h5'))):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables or lenet.h5 file is missing")

        if not self.keras_installed:
            unittest.TestCase.skipTest(self, "keras is not installed")
        
        # parameter for RNN layers
        rnn_size = 10
        feature_dim = 4

        # output classes
        output_dim = 29

        # maximum sequence length
        max_seq_len = 100        
        
        # define data specs needed to import Keras model weights
        tokensize = feature_dim

        inputs = []
        for fi in range(max_seq_len):
            for vi in range(tokensize):
                inputs.append('_f%d_v%d_' % (fi, vi))
        targets = ['y%d'%i for i in range(0, max_seq_len)]        
        
        data_spec = []
        data_spec.append(DataSpec(type_='NUMERICNOMINAL',
                                  layer='the_input',
                                  data=inputs, 
                                  numeric_nominal_parms=DataSpecNumNomOpts(length='_num_frames_',token_size=feature_dim)))
        data_spec.append(DataSpec(type_='NUMERICNOMINAL',
                                  layer='out', 
                                  data=targets, 
                                  nominals=targets, 
                                  numeric_nominal_parms=DataSpecNumNomOpts(length='ylen', token_size=1)))        
                                          
        # try all RNN model types
        for layer_type in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
            for bidirectional in [True, False]:
                model = define_keras_rnn_model(layer_type, bidirectional, rnn_size, feature_dim, output_dim)
                
                model_name = 'dlpy_model'
                model1, use_gpu = Model.from_keras_model(conn=self.s, 
                                                        keras_model=model,
                                                        max_num_frames=max_seq_len,
                                                        include_weights=True,
                                                        output_model_table=model_name)                                
        
                model1.print_summary()
                
                # try to load weights, but skip any GPU-based models because worker/soloist may not have GPU
                if os.path.isdir(self.data_dir) and (not use_gpu):
                    try:
                        copyfile(os.path.join(os.getcwd(),'dlpy_model_weights.kerasmodel.h5'),os.path.join(self.data_dir,'dlpy_model_weights.kerasmodel.h5'))
                        copy_success = True
                    except:
                        print('Unable to copy weights file, skipping test of attaching weights')
                        copy_success = False
                        
                    if copy_success:
                        model1.load_weights(path=os.path.join(self.data_dir,'dlpy_model_weights.kerasmodel.h5'),
                                            labels=False, use_gpu=use_gpu)
                        os.remove(os.path.join(self.data_dir,'dlpy_model_weights.kerasmodel.h5'))
                else:
                    print('GPU model, skipping test of attaching weights')

                if os.path.isfile(os.path.join(os.getcwd(),'dlpy_model_weights.kerasmodel.h5')):
                    os.remove(os.path.join(os.getcwd(),'dlpy_model_weights.kerasmodel.h5'))

                # clean up models
                del model
                del model1

                # clean up model table
                model_tbl_opts = input_table_check(model_name)
                self.s.table.droptable(quiet=True, **model_tbl_opts)

    def test_model_conversion3(self):
        '''
        Import CNN image classification model and override attributes
          - instantiate a Keras LeNet model and translate to DLPy/Viya model
            override CNN model attributes with RNN atttributes - never would be done
            in practice, just to verify that new attributes written
        NOTE: cannot attach weights unless both client and server share
              the same file system
        COVERAGE: from_keras_model(), load_weights() in network.py
                  keras_to_sas() in sas_keras_parse.py
                  write_keras_hdf5() in write_keras_model_parm.py
                  all functions in model_conversion_utils.py
                  CNN-related function in write_sas_code.py
        '''

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isfile(os.path.join(self.data_dir_local,'lenet.h5'))):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables or lenet.h5 file is missing")

        if self.keras_installed:
            from keras.models import Sequential
            from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
        else:
            unittest.TestCase.skipTest(self, "keras is not installed")

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
        model1,use_gpu = Model.from_keras_model(conn=self.s, keras_model=model, output_model_table=model_name,
                                                include_weights=True, scale=1.0/255.0,
                                                input_weights_file=os.path.join(self.data_dir_local,'lenet.h5'))  
        
        if os.path.isdir(self.data_dir):
            try:
                copyfile(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5'),os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'))
                copy_success = True
            except:
                print('Unable to copy weights file, skipping test of overriding attributes')
                copy_success = False

            if copy_success:
                self.s.table.addcaslib(activeonadd=False,datasource={'srctype':'path'},
                          name='MODEL_CONVERT',path=self.data_dir,subdirectories=True)

                model1.load_weights(path=os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'),
                                    labels=False, 
                                    use_gpu=use_gpu)

                os.remove(os.path.join(self.data_dir,'lenet_weights.kerasmodel.h5'))
                
                # parameter for (nonexistent) RNN layers
                rnn_size = 10
                feature_dim = 4

                # output classes
                output_dim = 29

                # maximum sequence length
                max_seq_len = 100        
                
                # define data specs needed to import Keras model weights
                tokensize = feature_dim

                inputs = []
                for fi in range(max_seq_len):
                    for vi in range(tokensize):
                        inputs.append('_f%d_v%d_' % (fi, vi))
                targets = ['y%d'%i for i in range(0, max_seq_len)]        
                
                data_spec = []
                data_spec.append(DataSpec(type_='NUMERICNOMINAL',
                                          layer=model.layers[0].name+"_input",
                                          data=inputs, 
                                          numeric_nominal_parms=DataSpecNumNomOpts(length='_num_frames_',token_size=feature_dim)))
                data_spec.append(DataSpec(type_='NUMERICNOMINAL',
                                          layer=model.layers[-1].name, 
                                          data=targets, 
                                          nominals=targets, 
                                          numeric_nominal_parms=DataSpecNumNomOpts(length='ylen', token_size=1)))                    
                
                # override model attributes
                from dlpy.attribute_utils import create_extended_attributes
                create_extended_attributes(self.s, model_name, model1.layers, data_spec)

        if os.path.isfile(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5')):
            os.remove(os.path.join(os.getcwd(),'lenet_weights.kerasmodel.h5'))

        # clean up model table
        model_tbl_opts = input_table_check(model_name)
        self.s.table.droptable(quiet=True, **model_tbl_opts)

        # clean up models
        del model
        del model1

    @classmethod
    def tearDownClass(cls):
    
        # drop action set
        cls.s.dropactionset(actionSet='deeplearn', _messagelevel='error')
    
        # tear down tests
        try:
            cls.s.terminate()
        except swat.SWATError:
            pass
        del cls.s
        swat.reset_option()
    