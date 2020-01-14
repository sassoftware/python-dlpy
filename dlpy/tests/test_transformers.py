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
from dlpy.utils import input_table_check, get_cas_host_type, DLPyError, caslibify
import pandas as pd
import unittest
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#from dlpy.transformers.bert_utils import bert_prepare_data
#from dlpy.transformers.bert_model import BERT_Model

class TestTransformers(unittest.TestCase):
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
        from transformers import BertTokenizer
        necessary_packages_installed = True
        sys.stderr = stderr
    except:
        necessary_packages_installed = False

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

    def test_transformers1(self):
        '''
        Prepare labeled data for single sentence BERT classification problem
        COVERAGE: bert_prepare_data() in bert_utils.py
                  class BertDMH() in bert_utils.py
        '''

        try:
            from dlpy.transformers.bert_utils import bert_prepare_data
            from dlpy.transformers.bert_model import BERT_Model
        except (ImportError, DLPyError) as e:
            unittest.TestCase.skipTest(self, "Unable to import from transformers. Please install it and try again.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isdir(self.data_dir_local)):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment "
                                             "variables or it does not exist.")

        if not self.necessary_packages_installed:
            unittest.TestCase.skipTest(self, "missing transformers package")
            
        if not os.path.isfile(os.path.join(self.data_dir_local,'imdb_master.csv')):
            unittest.TestCase.skipTest(self, "cannot locate imdb_master.csv in DLPY_DATA_DIR_LOCAL")

        from transformers import BertTokenizer
        model_name = 'bert-base-uncased'

        # instantiate BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name,cache_dir=self.data_dir_local)
        
        # read dataset for IMDB movie review sentiment classification
        reviews = pd.read_csv(os.path.join(self.data_dir_local,'imdb_master.csv'),
                              header=0,
                              names=['type', 'review','label','file'],
                              encoding='latin_1')

        input_label = 'review'       # input data is review text
        target_label = 'label'       # target data is sentiment label
        
        # extract "train" data
        t_idx1 = reviews['type'] == 'train'
        t_idx2 = reviews['label'] != 'unsup'
        inputs = reviews[t_idx1 & t_idx2][input_label].to_list()
        targets = reviews[t_idx1 & t_idx2][target_label].to_list()
        
        # limit the number of observations to 1000
        if len(inputs) > 1000:
            inputs = inputs[:1000]
            targets = targets[:1000]

        # create numeric target labels
        for ii,val in enumerate(targets):
            inputs[ii] = inputs[ii].replace("<br />","")
            if val == 'neg':
                targets[ii] = 1
            elif val == 'pos':
                targets[ii] = 2        
        
        # prepare data
        num_tgt_var, train = bert_prepare_data(self.s, 
                                               tokenizer, 
                                               128, 
                                               input_a=list(inputs), 
                                               target=list(targets), 
                                               segment_vocab_size=2,
                                               classification_problem=True)        
        
        # check for the existence of the training table
        res = self.s.retrieve('table.tableexists', _messagelevel='error', name=train)
        self.assertTrue(res['exists'] != 0,"Training table not created.")

        # ensure table has the proper number of columns
        res = self.s.retrieve('table.columninfo', _messagelevel='error', table=train)
        self.assertTrue(len(res['ColumnInfo']['Column'].to_list()) == 5,"Training table has extra/missing columns.")
        
        # clean up data table if it exists
        try:
            model_tbl_opts = input_table_check(train)
            self.s.table.droptable(quiet=True, **model_tbl_opts)
        except TypeError:
            self.assertTrue(False, "BERT data preparation failed")
        
        # clean up tokenizer
        del tokenizer

    def test_transformers2(self):
        '''
        Prepare labeled data for single sentence BERT regression problem
        COVERAGE: bert_prepare_data() in bert_utils.py
                  class BertDMH() in bert_utils.py
        '''

        try:
            from dlpy.transformers.bert_utils import bert_prepare_data
            from dlpy.transformers.bert_model import BERT_Model
        except (ImportError, DLPyError) as e:
            unittest.TestCase.skipTest(self, "Unable to import from transformers. Please install it and try again.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isdir(self.data_dir_local)):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment "
                                             "variables or it does not exist.")

        if not self.necessary_packages_installed:
            unittest.TestCase.skipTest(self, "missing transformers package")
            
        if not os.path.isfile(os.path.join(self.data_dir_local,'task1_training_edited.csv')):
            unittest.TestCase.skipTest(self, "cannot locate task1_training_edited.csv in DLPY_DATA_DIR_LOCAL")

        from transformers import BertTokenizer
        model_name = 'bert-base-uncased'

        # instantiate BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name,cache_dir=self.data_dir_local)
        
        # read regression data set
        reviews = pd.read_csv(os.path.join(self.data_dir_local,'task1_training_edited.csv'),
                              header=None,
                              names=['id','original','edit','grades','meanGrade'])

        inputs = reviews['original'].tolist()[1:]
        reviews['meanGrade'] = pd.to_numeric(reviews['meanGrade'], errors='coerce').fillna(0)
        targets = reviews['meanGrade'].tolist()[1:]
        for ii,val in enumerate(targets):
            targets[ii] = round(val)
        
        # limit the number of observations to 1000
        if len(inputs) > 1000:
            inputs = inputs[:1000]
            targets = targets[:1000]

        
        # prepare data
        num_tgt_var, train, valid = bert_prepare_data(self.s, 
                                                      tokenizer, 
                                                      128, 
                                                      input_a=list(inputs), 
                                                      target=list(targets),
                                                      segment_vocab_size=2,
                                                      train_fraction=0.8,
                                                      classification_problem=False)        
                
        # check for the existence of the training table
        res = self.s.retrieve('table.tableexists', _messagelevel='error', name=train)
        self.assertTrue(res['exists'] != 0,"Training table not created.")

        # ensure table has the proper number of columns
        res = self.s.retrieve('table.columninfo', _messagelevel='error', table=train)
        self.assertTrue(len(res['ColumnInfo']['Column'].to_list()) == 5,"Training table has extra/missing columns.")

        # check for the existence of the validation table
        res = self.s.retrieve('table.tableexists', _messagelevel='error', name=valid)
        self.assertTrue(res['exists'] != 0,"Validation table not created.")

        # ensure table has the proper number of columns
        res = self.s.retrieve('table.columninfo', _messagelevel='error', table=valid)
        self.assertTrue(len(res['ColumnInfo']['Column'].to_list()) == 5,"Validation table has extra/missing columns.")

        # clean up training table if it exists
        try:
            model_tbl_opts = input_table_check(train)
            self.s.table.droptable(quiet=True, **model_tbl_opts)
        except TypeError:
            self.assertTrue(False, "BERT data preparation failed")

        # clean up validation table if it exists
        try:
            model_tbl_opts = input_table_check(valid)
            self.s.table.droptable(quiet=True, **model_tbl_opts)
        except TypeError:
            self.assertTrue(False, "BERT data preparation failed")
        
        # clean up models
        del tokenizer

    def test_transformers3(self):
        '''
        Prepare test data (no labels) for two sentence BERT classification problem
        COVERAGE: bert_prepare_data() in bert_utils.py
                  class BertDMH() in bert_utils.py
        '''

        try:
            from dlpy.transformers.bert_utils import bert_prepare_data
            from dlpy.transformers.bert_model import BERT_Model
        except (ImportError, DLPyError) as e:
            unittest.TestCase.skipTest(self, "Unable to import from transformers. Please install it and try again.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isdir(self.data_dir_local)):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment "
                                             "variables or it does not exist.")

        if not self.necessary_packages_installed:
            unittest.TestCase.skipTest(self, "missing transformers package")
            
        if not os.path.isfile(os.path.join(self.data_dir_local,'qnli_train.tsv')):
            unittest.TestCase.skipTest(self, "cannot locate qnli_train.csv in DLPY_DATA_DIR_LOCAL")

        from transformers import BertTokenizer
        model_name = 'bert-base-uncased'

        # instantiate BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained(model_name,cache_dir=self.data_dir_local)
        
        # read QNLI dataset
        train_data = pd.read_csv('/dept/cas/DeepLearn/docair/glue/qnli/train.tsv',
                              header=0,
                              sep='\t',
                              error_bad_lines=False,
                              warn_bad_lines=False,
                              names=['index', 'question','sentence','label'])

        input_a_label = 'question'
        input_b_label = 'sentence'

        input_a = train_data[input_a_label].to_list()
        input_b = train_data[input_b_label].to_list()
                        
        # limit the number of observations to 1000
        if len(input_a) > 1000:
            input_a = input_a[:1000]
            input_b = input_b[:1000]
        
        # prepare data
        num_tgt_var, test = bert_prepare_data(self.s, 
                                              tokenizer, 
                                              128, 
                                              input_a=input_a,
                                              input_b=input_b,
                                              segment_vocab_size=2,
                                              classification_problem=True)        
                
        # check for the existence of the training table
        res = self.s.retrieve('table.tableexists', _messagelevel='error', name=test)
        self.assertTrue(res['exists'] != 0,"Test table not created.")

        # ensure table has the proper number of columns
        res = self.s.retrieve('table.columninfo', _messagelevel='error', table=test)
        self.assertTrue(len(res['ColumnInfo']['Column'].to_list()) == 3,"Test table has extra/missing columns.")

        # clean up data table if it exists
        try:
            model_tbl_opts = input_table_check(test)
            self.s.table.droptable(quiet=True, **model_tbl_opts)
        except TypeError:
            self.assertTrue(False, "BERT data preparation failed")
        
        # clean up tokenizer
        del tokenizer

    def test_transformers4(self):
        '''
        Load a base BERT model and add classification head.
        COVERAGE: BERT_Model() class in bert_model.py
                  all private class functions (e.g. _XXX) in bert_model.py
                  compile() in bert_model.py
                  load_weights() in bert_model.py
                  write_block_information() in bert_utils.py
                  get_data_spec() in bert_model.py
                  create_data_spec() in bert_utils.py
                  generate_target_var_names() in bert_utils.py
                  extract_pytorch_parms() in bert_utils.py
                  find_pytorch_tensor() in bert_utils.py
                  
        '''

        try:
            from dlpy.transformers.bert_utils import bert_prepare_data
            from dlpy.transformers.bert_model import BERT_Model
        except (ImportError, DLPyError) as e:
            unittest.TestCase.skipTest(self, "Unable to import from transformers. Please install it and try again.")

        model_name = 'bert-base-uncased'
        cache_dir = self.data_dir_local
        
        try:
            stderr = sys.stderr
            sys.stderr = open(os.devnull, 'w')
            import h5py
            h5py_installed = True
            sys.stderr = stderr
        except:
            h5py_installed = False

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        if (self.data_dir_local is None) or (not os.path.isdir(self.data_dir_local)):
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment "
                                             "variables or it does not exist.")

        if (not self.necessary_packages_installed) or (not h5py_installed):
            unittest.TestCase.skipTest(self, "missing transformers or h5py package")
            
        # test case parameters
        n_classes = 2
        num_encoder_layers = 2
        num_tgt_var = 1
        
        # instantiate BERT model
        bert = BERT_Model(self.s,
                          cache_dir,
                          model_name,
                          n_classes,
                          num_hidden_layers = num_encoder_layers,
                          verbose=False)
                          
        # compile model
        bert.compile(num_target_var=num_tgt_var)
        
        if not os.path.isfile(os.path.join(cache_dir,model_name+'.kerasmodel.h5')):
            assertTrue(False,"HDF5 file not written.")
            
        # check for the existence of the model table
        res = self.s.retrieve('table.tableexists', _messagelevel='error',
                              name=bert.model_name)
        self.assertTrue(res['exists'] != 0,"Model table not created.")
        
        # attempt to create CASLIB to cache directory
        try:
            caslib, extra_path, newlib = caslibify(self.s, cache_dir, task='save')
            do_load_weights = True
        except DLPyError:
            do_load_weights = False
            
        # attach model weights - skip if server unable to "see" cache directory
        if do_load_weights:
            bert.load_weights(os.path.join(cache_dir,model_name+'.kerasmodel.h5'), 
                              num_target_var=num_tgt_var, 
                              freeze_base_model=False)
                              
            # check for the existence of the weight table
            res = self.s.retrieve('table.tableexists', _messagelevel='error',
                                  name=bert.model_name+'_weights')
            self.assertTrue(res['exists'] != 0,"Weight table not created.")
                                                        
        # create data spec for model
        data_spec = bert.get_data_spec(num_tgt_var)
                                                            
        # drop table(s)
        try:
            model_tbl_opts = input_table_check(bert.model_name)
            self.s.table.droptable(quiet=True, **model_tbl_opts)
        except TypeError:
            self.assertTrue(False, "Unable to drop model table.")

        if do_load_weights:
            try:
                model_tbl_opts = input_table_check(bert.model_name+'_weights')
                self.s.table.droptable(quiet=True, **model_tbl_opts)
            except TypeError:
                self.assertTrue(False, "Unable to drop weight table.")
        
        # remove HDF5 file
        if os.path.isfile(os.path.join(cache_dir,model_name+'.kerasmodel.h5')):
            os.remove(os.path.join(cache_dir,model_name+'.kerasmodel.h5'))

        # clean up BERT model
        del bert

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
    
