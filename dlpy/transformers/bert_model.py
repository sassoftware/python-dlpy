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

''' BERT models '''

import os
import codecs
import collections
import numpy as np
import pandas as pd
from dlpy.utils import DLPyError, caslibify
from dlpy.model import Model, TextParms, Optimizer, AdamSolver
from dlpy.layers import Res, Dense, OutputLayer, LayerNormalization, MultiHeadAttention, Recurrent, InputLayer
from dlpy.transformers.bert_utils import BertDMH, generate_hdf5_dataset_name, find_pytorch_tensor, extract_pytorch_parms
from dlpy.transformers.bert_utils import create_data_spec, write_block_information, BertCommon, generate_target_var_names
    
try:    
    from transformers import BertTokenizer, BertModel, BertConfig
    from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
    from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
    
    from distutils.version import StrictVersion
    import pkg_resources
    import warnings
    if StrictVersion(pkg_resources.get_distribution("transformers").version) > '2.3.0':
        warn_message = ("You are using a version of the transformers package ("
                        + pkg_resources.get_distribution("transformers").version + 
                        ") that has not been tested for compatibility.  Unexpected behavior "
                        "may occur.")
        warnings.warn(warn_message,UserWarning)
except:
    raise DLPyError("Unable to import from transformers.  Please install "
                    "a supported version (2.3.0 or earlier) and try again.")

try:
    import h5py
except:
    raise DLPyError("Unable to import h5py.  Please install h5py and try again.")

class BERT_Model(Model):

    '''
    BERT_Model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS/Viya connection object.
    cache_dir: string
        Specifies the fully-qualified path of directory where HuggingFace BERT models are stored.
        Note that this directory must be accessible by the Viya client.  Viya uses a client-server
        architecture, and there may be cases where the client and the server do not share a common
        file system.
    name: string
        Specifies the name of the HuggingFace BERT model to load (e.g. bert-based-uncased)
        Default: bert-base-uncased
    name: n_classes
        Specifies the number of classes used for model fine-tuning.  Setting n_classes to
        1 sets up a regression problem.
        Default: None
    num_hidden_layers : int, optional
        Specifies the number of hidden layers to load from the BERT model.
        Default: None
    max_seq_len: int, optional
        Specifies the maximum sequence length in tokens.
        Default: 128    
    seed: int, optional
        Specifies the seed used for random number generation.
        Default: 987654321
    save_embedding: Boolean, optional
        Specfies whether to save a text version of embedding table.
        Default: False
    verbose: Boolean, optional
        Specifies whether to print informative notes and messages.
        Default: False
    Returns
    -------
    :class:`BERT_Model`

    '''

    def __init__(self, conn, cache_dir, name='bert-base-uncased', n_classes=None,
                 num_hidden_layers=None, max_seq_len=128, seed=987654321, 
                 save_embedding=False, verbose=False):

        self._base_name = name.lower()
        self._cache_dir = cache_dir
        self.model_type = 'RNN'
        self.seed = seed
        self._verbose = verbose
        self.conn = conn
        
        if (n_classes is None) or (n_classes <= 0):
            raise DLPyError('You must specify a valid number of classes')
        elif n_classes == 1:
            self._classification_problem = False
            self._n_class_levels = 1
            self.model_name = 'regression'
        else:
            self._classification_problem = True
            self._n_class_levels = n_classes
            self.model_name = 'classification'
       
        # verify cache directory exists
        if not os.path.isdir(cache_dir):
            raise DLPyError('You specified an invalid cache directory ' + cache_dir)
        elif not os.access(cache_dir, os.W_OK):
            raise DLPyError('You do not have permission to write to directory ' + cache_dir)
        
        # verify model type is supported
        hf_base_name = self._base_name.split('-')[0]
        if hf_base_name not in ['bert', 'roberta', 'distilbert', 'distilroberta']:
            raise DLPyError('You specified an unsupported model variant.'
                            'Only bert-*, roberta-*, and distil*-*'
                            'models are supported.')
                            
        if self._verbose:
            print("NOTE: loading base model " + self._base_name + " ...")
            
        # instantiate specified HuggingFace model and tokenizer
        if hf_base_name == 'bert':
            num_layers = BertConfig.from_pretrained(self._base_name,
                                                    cache_dir=cache_dir).to_dict()['num_hidden_layers']
        elif hf_base_name == 'roberta':
            num_layers = RobertaConfig.from_pretrained(self._base_name,
                                                       cache_dir=cache_dir).to_dict()['num_hidden_layers']
        else:   # all distil* models
            num_layers = DistilBertConfig.from_pretrained(self._base_name,
                                                          cache_dir=cache_dir).to_dict()['n_layers']
        
        if num_hidden_layers is None:
            num_hidden_layers = num_layers
        else:
            if num_hidden_layers > num_layers:
                raise DLPyError('You specified more hidden layers than are available in '
                                'the base model.')

        if hf_base_name == 'bert':
            self._base_model = BertModel.from_pretrained(self._base_name,
                                                         cache_dir=cache_dir,
                                                         num_hidden_layers=num_hidden_layers)
            self._tokenizer = BertTokenizer.from_pretrained(self._base_name)
            self._config = BertConfig.from_pretrained(self._base_name,
                                                      cache_dir=cache_dir,
                                                      num_hidden_layers=num_hidden_layers).to_dict()
        elif hf_base_name == 'roberta':
            self._base_model = RobertaModel.from_pretrained(self._base_name,
                                                            cache_dir=cache_dir,
                                                            num_hidden_layers=num_hidden_layers)
            self._tokenizer = RobertaTokenizer.from_pretrained(self._base_name)
            self._config = RobertaConfig.from_pretrained(self._base_name,
                                                         cache_dir=cache_dir,
                                                         num_hidden_layers=num_hidden_layers).to_dict()
        else:   # all distil* models
            self._base_model = DistilBertModel.from_pretrained(self._base_name,
                                                               cache_dir=cache_dir,
                                                               n_layers=num_hidden_layers)
            self._tokenizer = DistilBertTokenizer.from_pretrained(self._base_name)
            self._config = DistilBertConfig.from_pretrained(self._base_name,
                                                            cache_dir=cache_dir,
                                                            n_layers=num_hidden_layers).to_dict()
            
        # set model-specific quantitites
        self._set_bert_type_info(hf_base_name)
        
        if self._verbose:
            print("NOTE: base model " + self._base_name + " loaded.")
       
        # load embedding table (token | position | segment)
        self._load_embedding_table(save_embedding)
            
        if self._config['max_position_embeddings'] < max_seq_len:
            raise DLPyError('The specified maximum sequence length exceeds the maximum position embedding.')
        else:
            self._max_seq_len = max_seq_len
            
        # set input layers
        if self._use_segment_embedding:
            self.ds_layers = dict(token_input=BertCommon['layer_names']['token_input'],
                                  position_input=BertCommon['layer_names']['position_input'],
                                  segment_input=BertCommon['layer_names']['segment_input'])
        else:
            self.ds_layers = dict(token_input=BertCommon['layer_names']['token_input'],
                                  position_input=BertCommon['layer_names']['position_input'])
        
    def _set_bert_type_info(self, bert_variant_name):
        if bert_variant_name == 'bert':
            self._use_pooling_layer = True
            self._position_embedding_offset = 0
        elif bert_variant_name == 'roberta':
            self._use_pooling_layer = True
            self._position_embedding_offset = 2     # see comment in modeling_roberta.py, class RobertaEmbeddings
        elif 'distil' in bert_variant_name:
            # distilBERT needs to map some keys for consistency with base BERT model
            self._config['num_hidden_layers'] = self._config['n_layers']
            self._config['hidden_size'] = self._config['dim']
            self._config['hidden_act'] = self._config['activation']
            self._config['hidden_dropout_prob'] = self._config['dropout']
            self._config['num_attention_heads'] = self._config['n_heads']
            self._config['layer_norm_eps'] = 1e-12
            self._config['intermediate_size'] = self._config['hidden_dim']
            
            self._use_pooling_layer = False
            self._position_embedding_offset = 0
        else:
            raise DLPyError(bert_variant_name + " is not a supported model.")

        if 'type_vocab_size' in self._config:
            self._use_segment_embedding = True
        else:
            self._use_segment_embedding = False
            
        if self._config['num_hidden_layers'] > 0:
        
            # extract keywords for encoder block(s)
            layer_keys = []
            for param_tensor in self._base_model.state_dict():
                tmp = param_tensor.split('.')
                if all(c in tmp for c in ['0','weight']):
                    layer_keys.append(tmp[0:2]+tmp[3:-1])
                    
            # encoder block consists of:
            # 1) multi-head attention layer (3 dense subspace layers and one output dense layer)
            # 2) first residual layer (no parameters)
            # 3) first layernorm layer
            # 4) intermediate dense layer
            # 5) output dense layer
            # 6) second residual layer
            # 7) second layernorm layer
            #
            # NOTE: if the above changes then the following code AND _add_bert_encoding_layer
            #       must be changed
            self._encoding_blk_keys = [None]*BertCommon['n_sublayers_per_encoder']
            
            # multi-head attention keys
            self._encoding_blk_keys[0] = {'base':layer_keys[0][0:2], 'extra':[str_list[2:] for str_list in layer_keys[0:4]]}
            
            # first layernorm layer
            self._encoding_blk_keys[2] = {'base':layer_keys[4][0:2], 'extra':layer_keys[4][2:]}
            
            # intermediate dense layer
            self._encoding_blk_keys[3] = {'base':layer_keys[5][0:2], 'extra':layer_keys[5][2:]}

            # output dense layer
            self._encoding_blk_keys[4] = {'base':layer_keys[6][0:2], 'extra':layer_keys[6][2:]}

            # second layernorm layer
            self._encoding_blk_keys[6] = {'base':layer_keys[7][0:2], 'extra':layer_keys[7][2:]}
                            
    def _write_embedding_table(self):

        # embedding table read/written to cache directory
        embedding_file = os.path.join(self._cache_dir,self._base_name+'_embedding.txt')

        # create dictionary needed to convert list(s) to dataframe
        embed_dict = {}
        for ii, key in enumerate(self._embedding_var_names):
            embed_dict[key] = self._embedding_table[ii]
            
        # create dataframe
        df = pd.DataFrame(embed_dict)
        
        # write to file
        df.to_csv(embedding_file, sep=BertCommon['file_delimiter'], encoding='utf-8')
                
    def _load_embedding_table(self, save_embedding):

        # set up embedding table variable names and types
        self._embedding_var_names = ['term'] + [None] * self._config['hidden_size']
        self._embedding_var_types = ['VARCHAR'] + ['NUMERIC'] * self._config['hidden_size']
        for ii in range(0,self._config['hidden_size']):
            self._embedding_var_names[ii+1] = "_"+str(ii+1)+"_"                                    
    
        # check whether embedding table already constructed and saved
        embedding_file = os.path.join(self._cache_dir,self._base_name+'_embedding.txt')
        if os.path.isfile(embedding_file):
        
            if self._verbose:
                print("NOTE: loading embedding table ... ")
        
            # read embedding file to Pandas data frame
            df = pd.read_csv(embedding_file,
                             header=0,
                             sep=BertCommon['file_delimiter'],
                             names=self._embedding_var_names)
                             
            # convert dataframe to lists
            self._embedding_table = [None]*len(self._embedding_var_names)
            for ii, key in enumerate(self._embedding_var_names):
                self._embedding_table[ii] = df[key].to_list()

            if self._verbose:
                last_message = "NOTE: embedding table loaded."
        
        else:
                
            if self._verbose:
                print("NOTE: creating embedding table ... ")

            self._embedding_table = []

            # search model's state_dict for wordpiece embedding table
            for param_tensor in self._base_model.state_dict():
                tensor_shape = self._base_model.state_dict()[param_tensor].shape
                if tensor_shape == (self._config['vocab_size'],self._config['hidden_size']):
                    break
                
            token_embedding_table = self._base_model.state_dict()[param_tensor].numpy()

            # search model's state_dict for position embedding table
            for param_tensor in self._base_model.state_dict():
                tensor_shape = self._base_model.state_dict()[param_tensor].shape
                if tensor_shape == (self._config['max_position_embeddings'],self._config['hidden_size']):
                    break
                
            position_embedding_table = self._base_model.state_dict()[param_tensor].numpy()
                        
            # search model's state_dict for segment embedding table (does not occur in distilBERT* models)
            if self._use_segment_embedding:
                for param_tensor in self._base_model.state_dict():
                    tensor_shape = self._base_model.state_dict()[param_tensor].shape
                    if tensor_shape == (self._config['type_vocab_size'],self._config['hidden_size']):
                        break

                segment_embedding_table = self._base_model.state_dict()[param_tensor].numpy()
            else:
                segment_embedding_table = None
            
            # first column: embedding keys
            num_rows = self._config['vocab_size'] + self._config['max_position_embeddings'] - self._position_embedding_offset
            if segment_embedding_table is not None:
                num_rows += self._config['type_vocab_size']
                                
            # token embedding keys
            tmp_col = [None] * num_rows
            for r in range(0,self._config['vocab_size']):
                token = self._tokenizer.convert_ids_to_tokens(r)

                if token in BertCommon['reserved_names']:
                    raise DLPyError('Cannot creat embedding table, match for reserved name: ' + token)
                elif token in BertCommon['special_chars']:
                    tmp_col[r] = '['+token+']'
                else:
                    tmp_col[r] = token
                
            # position embedding keys
            table_offset = self._config['vocab_size']
            for r in range(0,self._config['max_position_embeddings']-self._position_embedding_offset):
                tmp_col[table_offset+r] = BertCommon['reserved_names']['position_prefix']+str(r)
            
            # segment embedding keys
            if self._use_segment_embedding:
                table_offset += (self._config['max_position_embeddings']-self._position_embedding_offset)
                for r in range(self._config['type_vocab_size']):
                    tmp_col[table_offset+r] = BertCommon['reserved_names']['segment_prefix']+str(r)
                
            self._embedding_table.append(tmp_col)
                
            # remaining columns: embedding vectors
            for c in range(0,self._config['hidden_size']):
            
                # token embedding vectors
                tmp_col = [None] * num_rows
                for r in range(0,self._config['vocab_size']):
                    tmp_col[r] = format(token_embedding_table[r][c], '.16f')

                # position embedding vectors
                row_offset = self._position_embedding_offset
                table_offset = self._config['vocab_size']
                for r in range(0,self._config['max_position_embeddings']-row_offset):
                    tmp_col[table_offset+r] = format(position_embedding_table[r+row_offset][c], '.16f')

                # segment embedding vectors
                if self._use_segment_embedding:
                    table_offset += (self._config['max_position_embeddings']-self._position_embedding_offset)
                    for r in range(0,self._config['type_vocab_size']):
                        tmp_col[table_offset+r] = format(segment_embedding_table[r][c], '.16f')
                                    
                self._embedding_table.append(tmp_col)
                
            if self._verbose:
                last_message = "NOTE: embedding table created and loaded."

            # write to file
            if save_embedding:
                self._write_embedding_table()
            
        # load embedding table to active CASLIB
        handler = BertDMH(self._embedding_table, self._embedding_var_names, self._embedding_var_types)
        self.conn.retrieve('table.addtable', _messagelevel='error',        
                           table=BertCommon['embedding_table_name'], 
                           replace=True, 
                           **handler.args.addtable)

        if self._verbose:
            print(last_message)

    def _find_layer_def(self, lname, layer_info):
        if isinstance(lname, list):
            names = lname
        else:
            names = [lname]
            
        ldefs = []
        for name in names:
            found_layer = False
            for key1 in layer_info.keys():
                # quit if layer found
                if found_layer:
                    break
                if isinstance(layer_info[key1], list):
                    for l_element in layer_info[key1]:
                        # quit if layer found
                        if found_layer:
                            break
                        for key2 in l_element.keys():
                            if l_element[key2]['name'] == name:
                                ldefs.append(l_element[key2]['ldef'])
                                found_layer = True
                                break
                else:
                    for key2 in layer_info[key1]:
                        if layer_info[key1][key2]['name'] == name:
                            ldefs.append(layer_info[key1][key2]['ldef'])
                            found_layer = True
                            break
            
            if not found_layer:
                raise DLPyError('Could not find definition for layer ' + name)
                
        return ldefs
        
    def _add_input_layers(self, width):
        # define layer information
        new_layer_info = collections.OrderedDict()
        
        # token input
        new_layer_info['token_input'] = dict(name = self.ds_layers['token_input'],
                                             type = BertCommon['layer_types']['noparms'], 
                                             dim = None,
                                             ldef = InputLayer(n_channels=1, width=width, height=1, name=self.ds_layers['token_input']))
        # add token_size member to InputLayer
        new_layer_info['token_input']['ldef']._token_size = self._config['hidden_size']
                                         
        # position input
        new_layer_info['position_input'] = dict(name = self.ds_layers['position_input'],
                                                type = BertCommon['layer_types']['noparms'], 
                                                dim = None,
                                                ldef = InputLayer(n_channels=1, width=width, height=1, name=self.ds_layers['position_input']))
        # add token_size member to InputLayer
        new_layer_info['position_input']['ldef']._token_size = self._config['hidden_size']
                                         
        # segment input
        if self._use_segment_embedding:
            new_layer_info['segment_input'] = dict(name = self.ds_layers['segment_input'],
                                                   type = BertCommon['layer_types']['noparms'], 
                                                   dim = None,
                                                   ldef = InputLayer(n_channels=1, width=width, height=1, name=self.ds_layers['segment_input']))
            # add token_size member to InputLayer
            new_layer_info['segment_input']['ldef']._token_size = self._config['hidden_size']
        
        return new_layer_info

    # based on BertEmbeddings() class from 
    # https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
    def _add_bert_embedding_layer(self, layer_name, layer_info):
        # find input layer(s)
        if self._use_segment_embedding:
            layer_list = [self.ds_layers['token_input'], self.ds_layers['position_input'], self.ds_layers['segment_input']]
        else:
            layer_list = [self.ds_layers['token_input'], self.ds_layers['position_input']]
        
        input_layers = self._find_layer_def(layer_list,
                                            layer_info)
                                             
        # need tensors for input layers
        for ii,ilayer in enumerate(input_layers):
            input_layers[ii] = input_layers[ii].input_tenor
        
        # define layer information
        new_layer_info = collections.OrderedDict()
        
        # residual layer to sum embeddings
        lname = layer_name + '_sum'
        new_layer_info['sum_layer'] = dict(name = lname, 
                                           type = BertCommon['layer_types']['noparms'], 
                                           dim = None,
                                           ldef = Res(name=lname,
                                                      act='identity')(input_layers))
                                          
        # layer normalization layer
        lname = layer_name + '_ln'
        new_layer_info['layer_norm_layer'] = dict(name = lname, 
                                                  type = BertCommon['layer_types']['layernorm'], 
                                                  dim = [self._config['hidden_size']],
                                                  ldef = LayerNormalization(name=lname, 
                                                                            act='identity',
                                                                            epsilon=self._config['layer_norm_eps'],
                                                                            token_size=self._config['hidden_size'],
                                                                            dropout=self._config['hidden_dropout_prob'])(new_layer_info['sum_layer']['ldef']))
                              
        return new_layer_info
                                 
    # based on BertLayer(), BertSelfAttention(), BertAttention(), BertIntermediate(), BertSelfOutput(), and BertOutput() 
    # classes from https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
    def _add_bert_encoding_layer(self, layer_base, src_layers, layer_info):

        # find input layer(s)
        input_layers = self._find_layer_def(src_layers, layer_info)        

        hidden_act = self._config['hidden_act']
        hidden_dropout_prob = self._config['hidden_dropout_prob']
        num_attention_heads = self._config['num_attention_heads']
        hidden_size = self._config['hidden_size']
        lnorm_eps = self._config['layer_norm_eps']
        intermediate_size = self._config['intermediate_size']
        
        # define layer info
        new_layer_info = collections.OrderedDict()
              
        # sublayer #1: multi-head attention 
        # NOTE: dropout performed on the Q*K^T matrix AND the output of the final dense layer
        lname = layer_base+'_mha'
        new_layer_info['mh_attention'] = dict(name = lname, 
                                              type = BertCommon['layer_types']['mhattention'], 
                                              dim = [self._config['hidden_size'], self._config['hidden_size']],
                                              ldef = MultiHeadAttention(name=lname, 
                                                                        n=hidden_size, 
                                                                        n_attn_heads=num_attention_heads, 
                                                                        act='auto', 
                                                                        dropout=hidden_dropout_prob)(input_layers))

        # sublayer #2: first residual 
        lname = layer_base+'_sum1'
        new_layer_info['sum1'] = dict(name = lname,
                                      type = BertCommon['layer_types']['noparms'], 
                                      dim = None,
                                      ldef = Res(name=lname,
                                                 act='identity')(input_layers+[new_layer_info['mh_attention']['ldef']]))

        # sublayer #3: first layer normalization 
        lname = layer_base+'_ln1'
        new_layer_info['layer_norm1'] = dict(name = lname,
                                             type = BertCommon['layer_types']['layernorm'], 
                                             dim = [self._config['hidden_size']],
                                             ldef = LayerNormalization(name=lname, 
                                                                       act='identity',
                                                                       token_size=self._config['hidden_size'],
                                                                       epsilon=lnorm_eps)(new_layer_info['sum1']['ldef']))
                                                                       
        # sublayer #4: intermediate fully-connected layer 
        lname = layer_base+'_intm_dense'
        new_layer_info['intermediate_fc'] = dict(name = lname, 
                                                 type = BertCommon['layer_types']['dense'], 
                                                 dim = [intermediate_size, self._config['hidden_size']],
                                                 ldef = Dense(name=lname,
                                                              act=hidden_act,
                                                              n=intermediate_size)(new_layer_info['layer_norm1']['ldef']))

        # sublayer #5: final fully-connected layer 
        lname = layer_base+'_final_dense'
        new_layer_info['final_fc'] = dict(name = lname, 
                                          type = BertCommon['layer_types']['dense'], 
                                          dim = [self._config['hidden_size'], intermediate_size],
                                          ldef = Dense(name=lname,
                                                       act='identity',
                                                       n=hidden_size,
                                                       dropout=hidden_dropout_prob)(new_layer_info['intermediate_fc']['ldef']))
                                          
        # sublayer #6: second residual 
        lname = layer_base+'_sum2'
        new_layer_info['sum2'] = dict(name = lname, 
                                      type = BertCommon['layer_types']['noparms'], 
                                      dim = None,
                                      ldef = Res(name=lname,
                                                 act='identity')([new_layer_info['layer_norm1']['ldef'], 
                                                                  new_layer_info['final_fc']['ldef']]))
        
        # sublayer #7: second layer normalization 
        lname = layer_base+'_ln2'
        new_layer_info['layer_norm2'] = dict(name = lname, 
                                             type = BertCommon['layer_types']['layernorm'], 
                                             dim = [self._config['hidden_size']],
                                             ldef = LayerNormalization(name=lname, 
                                                                       act='identity',
                                                                       token_size=self._config['hidden_size'],
                                                                       epsilon=lnorm_eps)(new_layer_info['sum2']['ldef']))

        if len(new_layer_info.keys()) != BertCommon['n_sublayers_per_encoder']:
            raise DLPyError("Number of sublayers for encoder block is incorrect.")
            
        return new_layer_info

    # based on BertPooler() class from
    # https://github.com/huggingface/pytorch-transformers/blob/master/pytorch_transformers/modeling_bert.py
    def _add_bert_pooling_layer(self, layer_base, src_layers, layer_info):

        # find input layer(s)
        input_layers = self._find_layer_def(src_layers, layer_info)        

        # define layer information
        new_layer_info = collections.OrderedDict()
        
        # fully-connected layer
        new_layer_info['pooling'] = dict(name = layer_base,
                                         type = BertCommon['layer_types']['dense'], 
                                         dim = [self._config['hidden_size'], self._config['hidden_size']],
                                         ldef = Dense(name=layer_base,
                                                       act='tanh',
                                                       n=self._config['hidden_size'])(input_layers))
                      
        return new_layer_info
                            
    def _add_task_head(self, src_layers, layer_info, num_target_var):
                
        # find input layer(s)
        input_layers = self._find_layer_def(src_layers, layer_info)        

        # define layer information
        new_layer_info = collections.OrderedDict()
                
        # add RNN layer:
        #
        #   -> extract classification token (first in sequence) for classification/regression tasks
        #   -> copy all tokens for sequence labeling tasks
        #
        # This layer is hidden from the end user as it acts as a pass-through
        # and is not trained so the parameters are fixed.
        #
        # NOTE: this layer uses IDENTITY initialization, so no parameters need to be
        #       specified
        self._rnn_layer = 'rnn_passthrough'
        if num_target_var == 1:
            rnn_output_type = 'ENCODING'
            rnn_reversed = True
        else:
            rnn_output_type = 'SAMELENGTH'
            rnn_reversed = False
            
        new_layer_info['rnn_layer'] = dict(name = self._rnn_layer,
                                           type = BertCommon['layer_types']['noparms'],
                                           dim = [self._config['hidden_size'], self._config['hidden_size']],
                                           ldef = Recurrent(name=self._rnn_layer,
                                                            n=self._config['hidden_size'],
                                                            rnn_type='RNN',
                                                            act='IDENTITY',
                                                            init='IDENTITY',
                                                            init_bias=0.0,
                                                            output_type=rnn_output_type,
                                                            reversed_=rnn_reversed)(input_layers))
                                           
        # determine activation function for final layer
        if self._classification_problem:
            activation = 'softmax'
        else:
            activation = 'identity'

        # define layer information
        layer_info = collections.OrderedDict()
        layer_info['task_layer'] = dict(name = self.ds_layers['task_layer'],
                                        type = BertCommon['layer_types']['dense'],
                                        dim = [self._config['hidden_size'], self._n_class_levels],
                                        ldef = OutputLayer(name=self.ds_layers['task_layer'],
                                                           n=self._n_class_levels,
                                                           act=activation)(new_layer_info['rnn_layer']['ldef']))
        
        # generate classification layer parameters (following HuggingFace)
        n_weights = self._config['hidden_size']*self._n_class_levels
        n_bias = self._n_class_levels
        np.random.seed(seed=self.seed)
        weights = np.random.normal(loc=0.0,
                                   scale=self._config['initializer_range'],
                                   size=(self._config['hidden_size'],self._n_class_levels))
        bias = np.zeros((self._n_class_levels),dtype='float')
        
        return layer_info, {self.ds_layers['task_layer']:{'weights':weights, 'bias':bias}}
        
    def _set_text_parameters(self):
        # set up parameters for text parsing
        self._text_parameters = TextParms(init_input_embeddings=BertCommon['embedding_table_name'])
        # indicate delimiter for internal tokenization algorithm
        self._text_parameters['textdelimiter'] = BertCommon['text_delimiter']
        
    def _from_huggingface_model(self):
        
        # verify key parameters
        if 'max_position_embeddings' not in self._config:
            raise DLPyError('Maximum position embedding is unspecified')
        elif 'hidden_size' not in self._config:
            raise DLPyError('Hidden size is unspecified')
                
        sas_layer_info = {}
        
        # input layers (tokens, position, and segments)
        sas_layer_info['input'] = self._add_input_layers(self._config['max_position_embeddings']*self._config['hidden_size'])
        
        # embedding layers (tokens, position, and segments)
        sas_layer_info['embedding'] = self._add_bert_embedding_layer('embedding', sas_layer_info)

        # encoding layers
        last_key = list(sas_layer_info['embedding'].keys())[-1]  # dictionary is ordered (OrderedDict), getting last layer added
        encoder_src_layer = sas_layer_info['embedding'][last_key]['name']
        sas_layer_info['encoder'] = []
        for lnum in range(self._config['num_hidden_layers']):
            sas_layer_info['encoder'].append(self._add_bert_encoding_layer('encoder'+str(lnum), 
                                                                           [encoder_src_layer],
                                                                           sas_layer_info))
            # dictionary is ordered (OrderedDict), getting last layer added
            last_key = list(sas_layer_info['encoder'][lnum].keys())[-1]
            encoder_src_layer = sas_layer_info['encoder'][lnum][last_key]['name']
            
        # pooling layer
        if self._use_pooling_layer:
            sas_layer_info['pooler'] = self._add_bert_pooling_layer('bert_pooling', encoder_src_layer, sas_layer_info)
        
        return sas_layer_info
    
    def _write_huggingface_bert_hdf5(self, sas_layer_info, task_parms):
        
        self._client_hdf5_file_name = os.path.join(self._cache_dir,self._base_name+'.kerasmodel.h5')
        
        if self._verbose:
            print('NOTE: HDF5 file is ' + self._client_hdf5_file_name)        

        # open output file
        try:
            f_out = h5py.File(self._client_hdf5_file_name, 'w')
        except IOError:
            raise DLPyError('The specified file cannot be written: ' + self._client_hdf5_file_name)

        try:
                     
            # create list of all layer names
            layer_names = []
            for key1 in sas_layer_info.keys():
                if isinstance(sas_layer_info[key1], list):
                    cur_layer_info = sas_layer_info[key1]
                else:
                    cur_layer_info = [sas_layer_info[key1]]
                for ii in range(len(cur_layer_info)):
                    for key2 in cur_layer_info[ii].keys():
                        if cur_layer_info[ii][key2]['type'] != BertCommon['layer_types']['noparms']:
                            layer_names.append(cur_layer_info[ii][key2]['name'])
                    
            f_out.attrs['layer_names'] = [l.encode('utf8') for l in layer_names]
                        
            # add parameters for embeddings
            if isinstance(sas_layer_info['embedding'], list):
                embedding_layer_info = sas_layer_info['embedding']
            else:
                embedding_layer_info = [sas_layer_info['embedding']]
                
            write_block_information(self._base_model, embedding_layer_info, ['embeddings', 'layernorm'], f_out)
                    
            # add parameters for encoder(s)
            if not isinstance(sas_layer_info['encoder'], list):
                raise DLPyError('Encoder layer information should be a list.')
            
            for ii in range(len(sas_layer_info['encoder'])):
                for jj,key in enumerate(sas_layer_info['encoder'][ii].keys()):
                    lname = sas_layer_info['encoder'][ii][key]['name']
                    ltype = sas_layer_info['encoder'][ii][key]['type']
                    ldim = sas_layer_info['encoder'][ii][key]['dim']
                    
                    if ltype != BertCommon['layer_types']['noparms']:
                        matval, vecval = extract_pytorch_parms(self._base_model, 
                                                               lname, 
                                                               ltype, 
                                                               ldim, 
                                                               self._encoding_blk_keys[jj]['base'] + [str(ii)],
                                                               self._encoding_blk_keys[jj]['extra'])

                        # there should be only one match for a given layer (aside from LAYERNORM layers)
                        if (len(matval) > 1) or ((vecval != None) and (len(vecval) > 1)):
                            if ltype == BertCommon['layer_types']['layernorm']:
                                if int(lname[-1]) == 1:
                                    index = 0
                                else:
                                    index = 1
                            else:
                                raise DLPyError('There were multiple Pytorch layers that matched layer ' + lname)
                        else:
                            index = 0
                        
                        g_out = f_out.create_group(lname)
                        new_weight_names = []

                        # save weights in format amenable to SAS
                        dset_name = generate_hdf5_dataset_name(lname, BertCommon['weight_index'])
                        new_weight_names.append(dset_name)
                        g_out.create_dataset(dset_name, data=matval[index])

                        # save bias in format amenable to SAS
                        if vecval is not None:
                            dset_name = generate_hdf5_dataset_name(lname, BertCommon['bias_index'])
                            new_weight_names.append(dset_name)
                            g_out.create_dataset(dset_name, data=vecval[index])

                        # update weight names
                        g_out.attrs['weight_names'] = new_weight_names            
            
            # add parameters for pooler
            if self._use_pooling_layer:
                if isinstance(sas_layer_info['pooler'], list):
                    pooling_layer_info = sas_layer_info['pooler']
                else:
                    pooling_layer_info = [sas_layer_info['pooler']]
                    
                write_block_information(self._base_model, pooling_layer_info, ['pooler','dense'], f_out)
                
            # add parameters for task-specific final layer(s)
            if 'task' in sas_layer_info:
                if isinstance(sas_layer_info['task'], list):
                    task_layer_info = sas_layer_info['task']
                else:
                    task_layer_info = [sas_layer_info['task']]
            
                for ii in range(len(task_layer_info)):
                    for key in task_layer_info[ii].keys():
                        lname = task_layer_info[ii][key]['name']
                        ltype = task_layer_info[ii][key]['type']
                        ldim = task_layer_info[ii][key]['dim']
                        
                        if ltype != BertCommon['layer_types']['noparms']:

                            g_out = f_out.create_group(lname)
                            new_weight_names = []

                            # save weights in format amenable to SAS
                            dset_name = generate_hdf5_dataset_name(lname, BertCommon['weight_index'])
                            new_weight_names.append(dset_name)
                            g_out.create_dataset(dset_name, data=task_parms[lname]['weights'])

                            # save bias in format amenable to SAS
                            if 'bias' in task_parms[lname]:
                                dset_name = generate_hdf5_dataset_name(lname, BertCommon['bias_index'])
                                new_weight_names.append(dset_name)
                                g_out.create_dataset(dset_name, data=task_parms[lname]['bias'])

                            # update weight names
                            g_out.attrs['weight_names'] = new_weight_names 
            else:
                raise DLPyError('No task-specific head attached to base model.')
                        
                        
        except ValueError as err_msg:
            print(err_msg)

        finally:
            # close files
            f_out.close()
                    
    def compile(self, num_target_var=1):
        '''
        Compiles a BERT model
            
        Parameters
        ----------
        num_target_var : int
            Specifies the number of target variables.  Typically 1 for 
            simple classification/regression tasks, and > 1 for sequence
            labeling tasks.
            Default: 1
            
        '''
        self.ds_layers['task_layer'] = BertCommon['layer_names']['task_layer']
        
        # extract base BERT model and convert to SAS model
        sas_layer_info = self._from_huggingface_model()
        
        # add task-specific head
        if self._use_pooling_layer:
            task_src_layer = sas_layer_info['pooler']['pooling']['name']
        else:
            if self._config['num_hidden_layers'] > 0:
                last_key = list(sas_layer_info['encoder'][self._config['num_hidden_layers']-1].keys())[-1]
                task_src_layer = sas_layer_info['encoder'][self._config['num_hidden_layers']-1][last_key]['name']
            else:
                last_key = list(sas_layer_info['embedding'].keys())[-1]
                task_src_layer = sas_layer_info['embedding'][last_key]['name']
            
        sas_layer_info['task'], task_parms = self._add_task_head(task_src_layer,
                                                                 sas_layer_info,
                                                                 num_target_var)

        # write HDF5 file to disk
        self._write_huggingface_bert_hdf5(sas_layer_info, task_parms)

        # collect input tensors
        input_layers = []
        for key in sas_layer_info['input'].keys():
            input_layers.append(sas_layer_info['input'][key]['ldef'].input_tenor)

        # collect output layers
        output_layers = []
        for key in sas_layer_info['task'].keys():
            if 'task_layer' in key:
                output_layers.append(sas_layer_info['task'][key]['ldef'])

        # compile the model
        super(BERT_Model, self).__init__(conn = self.conn,  
                                         inputs = input_layers, 
                                         outputs = output_layers,
                                         model_table=self.model_name,
                                         model_weights=self.model_name+'_weights')
                                         
        super(BERT_Model, self).compile()
                           
    def load_weights(self, 
                     path,
                     num_target_var=1,
                     freeze_base_model=False,
                     use_gpu=False,
                     last_frozen_layer=None):
        '''
        Load the weights from a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that
            contains the weight table.
        num_target_var : int
            Specifies the number of target variables.  Typically 1 for 
            simple classification/regression tasks, and > 1 for sequence
            labeling tasks.
            Default: 1            
        freeze_base_model : Boolean, optional
            Specifies whether to freeze the parameters of the base BERT model when performing fine-tuning training
            Default: False            
        use_gpu : Boolean, optional
            GPU processing of model required (or not)
            Default: False
        last_frozen_layer : string, optional
            Specifies the last layer in the model that is untrainable (frozen).  All previous model layers
            will be untrainable as well.  This parameter is ignored if freeze_base_model = False.
            Default : None

        Notes
        -----
        Currently support HDF5 and sashdat files.

        '''
        
        # data specs needed for importing a BERT model
        data_spec = self.get_data_spec(num_target_var)
                                                               
        # attach layer weights
        super(BERT_Model, self).load_weights(path, 
                                             data_spec=data_spec,
                                             use_gpu=use_gpu, 
                                             embedding_dim=self._config['hidden_size'])   

        # determine which layers to freeze
        self._freeze_layers = self._rnn_layer
            
        if freeze_base_model:
            self._freeze_base = True
            if last_frozen_layer is None:
                self._freeze_layers = self._rnn_layer
            else:
                self._freeze_layers = last_frozen_layer
            
        else:
            self._freeze_base = False
            self._freeze_layers = self._rnn_layer
                                                                                           
        # configure text parameters
        self._set_text_parameters()
        
        # configure optimizer parameters with defaults
        self.set_optimizer_parameters()
                                             
        
    def get_data_spec(self, num_target_var):
        '''
        Retrieves data specification needed for fine-tuning a BERT model

        Parameters
        ----------
        num_target_var: int
            Specifies the number of target variables.
        
        Returns
        -------
        :class:`DataSpec`
            
        '''
        
        return create_data_spec(self.ds_layers, self._classification_problem, num_target_var)
        
    def set_optimizer_parameters(self, beta1=0.9, beta2=0.999, learning_rate=3e-5, max_epochs=3, mini_batch_size=1):
        '''
        Sets optimizer parameters according to defaults in original BERT paper.

        Parameters
        ----------
        beta1 : double, optional
            Specifies the exponential decay rate for the first moment in
            the Adam learning algorithm.
            Default: 0.9
        beta2 : double, optional
            Specifies the exponential decay rate for the second moment in
            the Adam learning algorithm.
            Default: 0.999
        learning_rate : double, optional
            Specifies the learning rate.
            Default: 3e-5
        max_epochs : int, optional
            Specifies the maximum number of epochs.
            Default: 3
        mini_batch_size : int, optional
            Specifies the number of observations per thread in a mini-batch.
            Default: 1
            
        '''
    
        # the optimizer dictionary is set according to the parameter choices recommended
        # for fine-tuning in the original BERT paper
        if self._verbose:
            log_level = 3
        else:
            log_level = 0
            
        # determine what to train
        if self._freeze_base:
            freeze_layers_to = self._freeze_layers
            freeze_layers = None
        else:
            freeze_layers_to = None
            freeze_layers = [self._freeze_layers]
                
        # set optimizer parameters similar to settings in original BERT paper
        self._optimizer = Optimizer(algorithm=AdamSolver(beta1=beta1, beta2=beta2, learning_rate=learning_rate,
                                                         gamma=0.0, power=0.0),
                                    mini_batch_size=mini_batch_size, seed=self.seed, max_epochs=max_epochs,
                                    dropout_type='inverted', log_level=log_level, freeze_layers_to=freeze_layers_to,
                                    freeze_layers=freeze_layers)
                                   
    def get_tokenizer(self):
        '''
        Retrieves _tokenizer used by BERT model
        
        Returns
        -------
        :class:`BertTokenizer`
            
        '''
        
        return self._tokenizer

    def get_max_sequence_length(self):
        '''
        Extracts the maximum sequence length supported by BERT model
        
        Returns
        -------
        int - maximum sequence length supported by BERT model
            
        '''
        
        return self._max_seq_len

    def get_problem_type(self):
        '''
        Extracts BERT model problem configuration (classification or regression)
        
        Returns
        -------
        Boolean - classification problem (True) or regression problem (False)
            
        '''
        
        return self._classification_problem

    def get_hdf5_file_name(self):
        '''
        Extracts name of HDF5 file (client side) that stores BERT model parameters
        
        Returns
        -------
        string - fully qualified name of HDF5 parameter file
            
        '''
        
        return self._client_hdf5_file_name

    def get_optimizer(self):
        '''
        Extracts optimizer object configured for fine-tuning BERT model
        
        Returns
        -------
        :class:`Optimizer`
            
        '''
        
        return self._optimizer

    def get_text_parameters(self):
        '''
        Extracts TextParms object configured for BERT embedding and parsing
        tokenized text data
        
        Returns
        -------
        :class:`TextParms`
            
        '''
        
        return self._text_parameters

    def get_segment_size(self):
        '''
        Determines the number of supported segments from the 
        model configuration
        
        Returns
        -------
        int
            
        '''

        if 'type_vocab_size' in self._config:
            return self._config['type_vocab_size']
        else:
            return 0
