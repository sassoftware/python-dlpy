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

import os
import sys
import codecs
import random
import numpy as np
from numbers import Number
import swat.cas.datamsghandlers as dmh
from dlpy.utils import DLPyError, caslibify
from dlpy.model import DataSpec, DataSpecNumNomOpts

# constants/reserved values
BertCommon = {}
BertCommon['file_delimiter'] = '\t'
BertCommon['text_delimiter'] = ' '
BertCommon['special_chars'] = ["\"",'\'']
BertCommon['reserved_names'] = dict(position_prefix='position_', segment_prefix='segment_')
BertCommon['embedding_table_name'] = "bert_embedding"
BertCommon['layer_types'] = dict(layernorm = 1, mhattention = 2, dense = 3, noparms = 4)
BertCommon['variable_names'] = dict(token_var='_tokens_', position_var='_position_', segment_var='_segment_', 
                                    target_var='_target_', target_len_var='_target_length_', weight_var='_weight_')
BertCommon['layer_names'] = dict(token_input='token_input', position_input='position_input', segment_input='segment_input', 
                                 task_layer='outlayer')
BertCommon['weight_index'] = 0
BertCommon['bias_index'] = 1
BertCommon['n_sublayers_per_encoder'] = 7

''' Utilities used to build and manage BERT models '''

class BertDMH(dmh.CASDataMsgHandler):

    '''
    BertDMH

    Create data message handler for a table with mixed varchar and double columns
    Based on example at 
    https://sassoftware.github.io/python-swat/generated/swat.cas.datamsghandlers.CASDataMsgHandler.html#swat.cas.datamsghandlers.CASDataMsgHandler

    Parameters
    ----------
    data_array : list of lists
        Data table to load as CAS table.
    variable_names: list
        Text strings representing variable names
    variable_types: list
        Text strings representing variable types (NUMERIC or VARCHAR)
    Returns
    -------
    :class:`cas.DataMsgHandlers`

    '''

    def __init__(self, data_array, var_names, var_types):                    
        vars = [None] * len(data_array)
        offset = 0
        for ii in range(len(data_array)):  
            if var_types[ii] == 'NUMERIC':
                vars[ii] = dict(name=var_names[ii],
                                 label=var_names[ii], 
                                 length=8, 
                                 type='sas', 
                                 rtype='numeric', 
                                 offset=offset)
                offset = offset + 8
            elif var_types[ii] == 'VARCHAR':
                vars[ii] = dict(name=var_names[ii], 
                                 label=var_names[ii], 
                                 length=16,
                                 type='varchar', 
                                 rtype='char', 
                                 offset=offset)
                offset = offset + 16
            else:
                raise DLPyError("Unknown variable type " + var_types[ii] + " given.")
            
        self.data = [None] * len(data_array[0])
        tmp = [None] * len(data_array)
        for rr in range(len(data_array[0])):
            for cc in range(len(data_array)):
                tmp[cc] = data_array[cc][rr]
            self.data[rr] = tuple(tmp)

        super(BertDMH, self).__init__(vars)

    def getrow(self, row):
        try:
            return self.data[row]
        except IndexError:
            return

def generate_hdf5_dataset_name(lname, index):
    '''
    Generate data set names consistent with names used by Keras models

    Parameters
    ----------
    lname : string
       Layer name

    Returns
    -------
    UTF-8 encoded data set name

    '''
    template_names = ['kernel:0', 'bias:0']

    dataset_name = lname + '/' + template_names[index]
    return dataset_name.encode('utf8')
    
def find_pytorch_tensor(pymodel, keywords, param_dim):
    '''
    Find PyTorch tensor in state dictionary

    Parameters
    ----------
    pymodel : PyTorch model
        Specifies the Pytorch model object.
    keywords : string
        Specifies the keywords to search for in the model dictionary.
    param_dim: list of int
        Specifies the dimensions of a tensor.

    Returns
    -------
    PyTorch parameter name, PyTorch tensor

    '''

    parm_name = []
    parm_tensor = []
    
    keylist = [x.lower() for x in keywords]

    for param_tensor in pymodel.state_dict():
        param_tensor_keylist = [x.lower() for x in param_tensor.split('.')]
        if all(c in param_tensor_keylist for c in keylist):
            # name matches, make sure dimensions do as well
            if len(param_dim) == len(pymodel.state_dict()[param_tensor].shape):
                dim_match = True
                for ii in range(len(param_dim)):
                    if param_dim[ii] != pymodel.state_dict()[param_tensor].shape[ii]:
                        dim_match = False
                        break
                
                if dim_match:
                    parm_name.append(param_tensor)
                    parm_tensor.append(pymodel.state_dict()[param_tensor].numpy())
                
    if not parm_name:
        return None, None
    else:
        return parm_name, parm_tensor

def extract_pytorch_parms(pymodel, layer_name, layer_type, layer_dim, layer_keywords, extra_keywords=None):
    '''
    Extract correct tensor(s) from a PyTorch model state dictionary

    Parameters
    ----------
    pymodel : PyTorch model
        Specifies the PyTorch model object.
    layer_name: string
        Specifies the PyTorch layer name.
    layer_type: int
        Specifyies the layer type (see BertCommon for layer types).
    layer_dim: list of int
        Specifies the dimensions of a tensor.
    layer_keywords : list of strings
        Specifies the keywords to search for in the model dictionary.
    extra_keywords : list of strings or None, optional
        Specifies the extra keywords for a multi-head attention layer.
        This is mandatory for multi-head attention and any other layer(s)
        where there could be ambiguity between two layers with the same
        type.
        Default : None

    Returns
    -------
    weight, bias parameters

    '''
    
    ptensor_wgt = None
    ptensor_bias = None
    
    if layer_type == BertCommon['layer_types']['noparms']:
        pass
    elif layer_type == BertCommon['layer_types']['layernorm']:
        if extra_keywords is None:
            key_list = layer_keywords+['weight']
        else:
            key_list = layer_keywords+extra_keywords+['weight']
            
        # weights
        pname, ptensor_wgt = find_pytorch_tensor(pymodel, key_list, layer_dim)
        if pname == None:
            raise DLPyError('Cannot find weights for layer ' + layer_name)
        # bias
        key_list[-1] = 'bias'
        pname, ptensor_bias = find_pytorch_tensor(pymodel, key_list, layer_dim)
        if pname == None:
            print('NOTE: No bias for layer ' + layer_name)
    elif layer_type == BertCommon['layer_types']['dense']:
        if extra_keywords is None:
            key_list = layer_keywords+['weight']
        else:
            key_list = layer_keywords+extra_keywords+['weight']

        # weights
        pname, ptensor_wgt = find_pytorch_tensor(pymodel, key_list, layer_dim)
        if pname == None:
            raise DLPyError('Cannot find weights for layer ' + layer_name)
        # bias
        # NOTE: bias name and dimensions not unique in attention layer so construct bias tensor name from
        # weight tensor name
        bias_str = pname[0].replace('weight','bias')
        if bias_str in pymodel.state_dict():
            ptensor_bias = [pymodel.state_dict()[bias_str].numpy()]
        else:
            print('NOTE: No bias for layer ' + layer_name)
    elif layer_type == BertCommon['layer_types']['mhattention']:

        # NOTE: for affine transformations, Pytorch uses a linear layer that implements 
        # the following operation
        #
        # y = x*A^T + b
        #
        # where A is stored in the form (output_dimension, input_dimension) in the state dictionary
        #
        # SAS Deep Learning implements an affine transformation as follows
        #
        # y = A*x + b
        #
        # where A is stored in the form (output_dimension, input_dimension) in column-major order.
        #
        # For dense (fully-connected) layers, the key is the A matrix which is stored identically
        # in both cases so it can be imported directly without any manipulation.  The Pytorch version of
        # Multi-head attention uses several linear layers in the implementation.  The SAS Deep Learning 
        # version is self-contained, and implements something like
        #
        # y = x*A + b
        #
        # for these linear layers.  This means that the A matrices used by the SAS Deep Learning version of 
        # multi-head attention must be transposed before importing.
    
        for ii, mha_keys in enumerate(extra_keywords):
                
            # weights
            pname, tmp_wgt = find_pytorch_tensor(pymodel, layer_keywords+mha_keys+['weight'], layer_dim)
            if pname == None:
                raise DLPyError('Cannot find ' + str(mha_keys) + ' weights for layer ' + layer_name)
            else:
                if len(tmp_wgt) > 1:
                    raise DLPyError('Multiple matches for ' + str(mha_keys) + ' weights for layer ' + layer_name)
                else:
                    if ii == 0:
                        tensor_wgt = np.transpose(tmp_wgt[0].copy())
                    else:
                        tensor_wgt = np.concatenate((tensor_wgt, np.transpose(tmp_wgt[0])), axis=1)
                        
            # bias
            pname, tmp_bias = find_pytorch_tensor(pymodel, layer_keywords+mha_keys+['bias'], [layer_dim[0]])
            if pname == None:
                print('NOTE: No ' + str(mha_keys) + ' bias for layer ' + layer_name)
            else:
                if 'tensor_bias' in locals():
                    tensor_bias = np.concatenate((tensor_bias, tmp_bias[0]))
                else:
                    tensor_bias = tmp_bias[0].copy()
                
        ptensor_wgt = [tensor_wgt]
        if 'tensor_bias' in locals():
            ptensor_bias = [tensor_bias]
                
    else:
        raise DLPyError('Layer ' + layer_name + ' is an unsupported layer type')
        
    return ptensor_wgt, ptensor_bias

def generate_target_var_names(variables, seq_len):
    '''
    Generate target variable names when there are multiple targets.

    Parameters
    ----------
    variables : dictionary
        Specifies the variables bound to input/task layers.
    seq_len: int
        Specifies the worst-case target sequence length.

    Returns
    -------
    list of target variable names

    '''
    return [variables['target_var']+str(ii)+'_' for ii in range(seq_len)]
    
def create_data_spec(layers, classification_problem, max_seq_len):
    '''
    Create data specification corresponding to BERT model

    Parameters
    ----------
    layers: dictionary
        Specifies the input/task layers.
    classification_problem: boolean
        Specifies whether the problem is classfication (True) or regression (False).
    max_seq_len: int
        Specifies the maximum target sequence length.

    Returns
    -------
    data specification

    '''

    # tokenization, position, and segment embedding
    tokens = [BertCommon['variable_names']['token_var']]
    position = [BertCommon['variable_names']['position_var']]
    segment = [BertCommon['variable_names']['segment_var']]
            
    # target variables
    target = generate_target_var_names(BertCommon['variable_names'],max_seq_len)
    nominals = generate_target_var_names(BertCommon['variable_names'],max_seq_len)

    data_spec = []
    data_spec.append(DataSpec(type_='TEXT',
                              layer=layers['token_input'],
                              data=tokens))
    data_spec.append(DataSpec(type_='TEXT',
                              layer=layers['position_input'],
                              data=position))
    if 'segment_input' in layers:
        data_spec.append(DataSpec(type_='TEXT',
                                  layer=layers['segment_input'],
                                  data=segment))

    if classification_problem:
        data_spec.append(DataSpec(type_='NUMNOM',
                                  layer=layers['task_layer'],
                                  data=target,
                                  nominals=nominals,
                                  numeric_nominal_parms=DataSpecNumNomOpts(length=BertCommon['variable_names']['target_len_var'], 
                                                                           token_size=1)))        
    else:
        data_spec.append(DataSpec(type_='NUMNOM',
                                  layer=layers['task_layer'],
                                  data=target,
                                  numeric_nominal_parms=DataSpecNumNomOpts(length=BertCommon['variable_names']['target_len_var'], 
                                                                           token_size=1)))        
                            
    return data_spec

def write_block_information(pymodel, layer_info, keywords, f_out):
    '''
    Write information for a block of layers to an HDF5 file

    Parameters
    ----------
    pymodel : PyTorch model
        Specifies the Pytorch model object.
    layer_info : list of dictionaries
        Specifies a list of dictionaries - each dictionary entry defines a layer in the model.
    keywords : list of strings
        Specifies the keywords to search for in the PyTorch model dictionary.
    f_out: file handle
        Specifies the HDF5 file handle.

    Returns
    -------
    PyTorch parameter name, PyTorch tensor

    '''

    if isinstance(keywords, list):
        key_list = keywords
    else:
        key_list = [keywords]
        
    for ii in range(len(layer_info)):
        for key in layer_info[ii].keys():
            lname = layer_info[ii][key]['name']
            ltype = layer_info[ii][key]['type']
            ldim = layer_info[ii][key]['dim']
            
            if ltype != BertCommon['layer_types']['noparms']:
                matval, vecval = extract_pytorch_parms(pymodel, lname, ltype, ldim, key_list)
                
                # there should be only one match for a given layer
                if (len(matval) > 1) or ((vecval != None) and (len(vecval) > 1)):
                    raise DLPyError('There were multiple Pytorch layers that matched layer ' + lname)
                
                g_out = f_out.create_group(lname)
                new_weight_names = []

                # save weights in format amenable to SAS
                dset_name = generate_hdf5_dataset_name(lname, BertCommon['weight_index'])
                new_weight_names.append(dset_name)
                g_out.create_dataset(dset_name, data=matval[0])

                # save bias in format amenable to SAS
                if vecval is not None:
                    dset_name = generate_hdf5_dataset_name(lname, BertCommon['bias_index'])
                    new_weight_names.append(dset_name)
                    g_out.create_dataset(dset_name, data=vecval[0])
                    
                # update weight names
                g_out.attrs['weight_names'] = new_weight_names            

def bert_prepare_data(conn, tokenizer, max_seq_len, input_a, segment_vocab_size=None, input_b=None, 
                      target=None, obs_weight=None, extra_var=None, neutral_label=None, 
                      train_fraction=None, classification_problem=True, seed=777777777,
                      verbose=False):
    '''
    Prepare data for a BERT model variant

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection
    tokenizer : :class:PreTrainedTokenizer object
        Specifies the tokenizer.
    max_seq_len: int
        Specifies the maximum sequence length (maximum number of tokens).
    input_a : list of strings
        Specifies the text data for a single segment task.
    segment_vocab_size : int
        Specifies the segment vocabulary size.  The value should be
        one of 0 for DistilBERT, 1 for RoBERTa, or 2 for BERT.
        Default: None
    input_b : list of strings, optional
        Specifies the text data for a two segment task.
        Default: None
    target: list or list of lists, optional
        Specifies the target data.  Target data must be a numeric type.
        This means that nominal values must be translated to integer class levels.
        Default: None
    obs_weight: list of float/integers
        Specifies the observation weights.
        Default: None
    extra_var: list of dictionaries
        Specifies the extra variable(s) to include in the Viya table(s).
        Each dictionary in the list must have the following keys
            name: string, specifies the name of the extra variable
            values: list, specifies the variable values
            type: string, must be either VARCHAR for characer values or NUMERIC for numeric values
        Default: None
    neutral_label: string, optional
        Specifies the "don't care" or neutral target label for multi-target classification tasks.
        This is not optional if target is a list of lists.
        Default: None
    train_fraction: float, optional
        Specifies the fraction of the data used for training.  Must be between 0.0 and 1.0.
        Default: None
    classification_problem: boolean, optional
        Specifies whether the data is for a classification or regression problem.
        Default: True
    seed: int, optional
        Specifies the seed to use for the random number generator for splitting data into
        train and test data sets.
        Default: 777777777
    verbose: boolean, optional
        Specifies whether progress messages and summary statistics are displayed.
        Default: False
        
    Returns
    -------
     -> number of target variables (if target specified) or None.
     -> if train fraction specified : names of the Viya tables that hold the training and test/validation data sets 
        otherwise : name of the data set
        
    '''    

    # define input variables
    ds_vars = dict(token_var=BertCommon['variable_names']['token_var'], 
                   position_var=BertCommon['variable_names']['position_var'], 
                   segment_var=BertCommon['variable_names']['segment_var'])
    
    # error checking 
    if not isinstance(input_a, list):
        raise DLPyError('Input A must be a list')

    if input_b is not None:
        if not isinstance(input_b, list):
            raise DLPyError('Input B must be a list')
            
        if len(input_a) != len(input_b):
            raise DLPyError("Mismatch in lengths of input A and input B lists")
    
    if target is not None:
        if not isinstance(target, list):
            raise DLPyError('Target must be a list')

        if len(input_a) != len(target):
            raise DLPyError("Mismatch in lengths of input A and target lists")
            
        # target variable and length variable
        ds_vars['target_var'] = BertCommon['variable_names']['target_var']
        ds_vars['target_len_var'] = BertCommon['variable_names']['target_len_var']

    if obs_weight is not None:
        if not isinstance(obs_weight, list):
            raise DLPyError('Observation weights must be a list')

        if len(input_a) != len(obs_weight):
            raise DLPyError("Mismatch in lengths of input A and observation weight lists")
            
        if target is None:
            raise DLPyError("Weight specified without target variable.")
            
        # weight variable
        ds_vars['weight_var'] = BertCommon['variable_names']['weight_var']

    if extra_var is not None:
        extra_var_names = [None]*len(extra_var)
        extra_var_types = [None]*len(extra_var)
        
        if not isinstance(extra_var, list):
            raise DLPyError('Extra variables must be a list')
            
        for ii,ev_dict in enumerate(extra_var):
            if not isinstance(ev_dict, dict):
                raise DLPyError('Argument extra_var must be a list of dictionaries')
                
            if 'name' in ev_dict:
                extra_var_names[ii] = ev_dict['name']
            else:
                raise DLPyError('extra_var[' + str(ii) + '] missing "name" key.')
                
            if ('type' in ev_dict) and (ev_dict['type'].upper() in ['VARCHAR','NUMERIC']):
                extra_var_types[ii] = ev_dict['type'].upper()
            else:
                raise DLPyError('extra_var[' + str(ii) + '] missing "type" key, or an invalid type was specified.')
            
            if ('values' not in ev_dict) or (not (isinstance(ev_dict['values'], list) and (len(input_a) == len(ev_dict['values'])))):
                raise DLPyError('extra_var[' + str(ii) + '] missing "values" key, the values are not a list object, '
                                'or there is a mismatch in lengths of input A and values lists.')
                                
    else:
        extra_var_names = None
        extra_var_types = None
                    
    if (train_fraction is not None) and ((train_fraction < 0.0) or (train_fraction > 1.0)):
        raise DLPyError('train_fraction must be between 0 and 1')
        
    if segment_vocab_size is None:
        raise DLPyError("You must specify a segment vocabulary size.  See the Bert model "
                        "configuration object (e.g. BertConfig['type_vocab_size'] for the "
                        "correct value.")
    else:
        if segment_vocab_size not in [0, 1, 2]:
            raise DLPyError('Vocabulary size ' + str(segment_vocab_size) + ' is invalid. '
                            'The value must be 0, 1, or 2.')
                    
    # initialize lists
    token_strings = [None] * len(input_a)
    position_strings = [None] * len(input_a)
    if segment_vocab_size > 0:
        segment_strings = [None] * len(input_a)
    if target is not None:
        target_array = [None] * len(input_a)
        tgtlen_array = [None] * len(input_a)
    if obs_weight is not None:
        weight_array = [None] * len(input_a)
    if extra_var is not None:
        extra_var_array = [None] * len(input_a)
    
    num_truncated = 0
    obs_idx = 0
    ten_percent = int(0.1*len(input_a))
    multiple_targets = False
    for ii,txt_a in enumerate(input_a):
    
        # provide feedback
        if verbose:
            if (ii > 0) and (ii % ten_percent == 0):
                print("NOTE: " + str(int(round(ii*100.0/len(input_a)))) + "% of the observations tokenized.")
                
        # simple data cleaning, skip observations where input A is invalid
        if len(txt_a) == 0:
            continue
        else:
            txt_a_untok = txt_a
            txt_a = tokenizer.tokenize(txt_a)
            txt_a = txt_a[:min([max_seq_len,len(txt_a)])]   # NOTE: this suppresses an unnecessary logger warning
    
        # simple data cleaning, skip observations where input B is invalid
        if input_b is not None:
            txt_b = input_b[ii]
            txt_b_untok = txt_b
            if len(txt_b) == 0:
                continue
            else:
                txt_b = tokenizer.tokenize(txt_b)
                txt_b = txt_b[:min([max_seq_len,len(txt_b)])]   # NOTE: this supresses an unnecessary logger warning
        else:
            txt_b = None
            txt_b_untok = None
            
        # simple data cleaning, skip observations where target is invalid (i.e. not numeric data)
        if target is not None:
            cur_tgt = target[ii]
            
            if isinstance(cur_tgt, list):
                tst_val = cur_tgt[0]
            else:
                tst_val = cur_tgt
            
            if not isinstance(tst_val, Number):
                continue
        else:
            cur_tgt = None

        # observation weight
        if obs_weight is not None:
            cur_wgt = obs_weight[ii]
            if not isinstance(cur_wgt, Number):
                raise DLPyError('Observation weights must be a numeric type.')
        else:
            cur_wgt = None
            
        # extra variable(s)
        if extra_var is not None:
            cur_extra_var = [None]*len(extra_var)
            for jj,ev_dict in enumerate(extra_var):
                cur_extra_var[jj] = ev_dict['values'][ii]
        else:
            cur_extra_var = None
        
        # tokenize text
        txt_encoding = tokenizer.encode_plus(txt_a,
                                             text_pair=txt_b,
                                             add_special_tokens=True,
                                             return_special_tokens_mask=True,
                                             max_length=max_seq_len)
        tmp_tokenized_text = tokenizer.convert_ids_to_tokens(txt_encoding['input_ids'])
        
        # set segment ID
        if segment_vocab_size == 2:
            seg_idx = txt_encoding['token_type_ids']
        elif segment_vocab_size == 1:
            seg_idx = [0]*len(tmp_tokenized_text)
        else:
            seg_idx = None
            
        # check for truncated sequence(s)
        if 'num_truncated_tokens' in txt_encoding:
            num_truncated += 1
                                
        # tokenization error-checking
        num_tokens = len(tmp_tokenized_text)
        tokenized_text = [None] * num_tokens
        for jj in range(num_tokens):
                
            if tmp_tokenized_text[jj] in BertCommon['reserved_names']:
                raise DLPyError('Match for reserved names: ' + tmp_tokenized_text[jj])
            elif tmp_tokenized_text[jj] in BertCommon['special_chars']:
                tokenized_text[jj] = '['+tmp_tokenized_text[jj]+']'
            else:
                tokenized_text[jj] = tmp_tokenized_text[jj]
                
        # verify targets match inputs for sequence labeling tasks (assume single segment only for now)
        if isinstance(cur_tgt, list):
            multiple_targets = True
            
            if neutral_label is None:
                raise DLPyError("Neutral label must be specified for sequence labeling tasks.")
                                
            if txt_b_untok is None:
                num_words = len(txt_a_untok.split(BertCommon['text_delimiter']))
            else:
                num_words = (len(txt_a_untok.split(BertCommon['text_delimiter'])) +
                             len(txt_b_untok.split(BertCommon['text_delimiter'])))
                
            num_tgts = len(cur_tgt)
            if num_words != num_tgts:
                raise DLPyError("Mismatch in length of input/target for observation " + str(ii))
            
            # tokenization adds special tokens and may split words into multiple tokens.  Add
            # neutral labels for special tokens and repeat target labels for words split by
            # tokenization.
            new_tgt = [neutral_label if mask == 1 else None for mask in txt_encoding['special_tokens_mask']]
            txt_words = txt_a_untok.split(BertCommon['text_delimiter'])
            if txt_b_untok is not None:
                txt_words += txt_b_untok.split(BertCommon['text_delimiter'])
            
            idx = 0
            for cur_word,cur_label in zip(txt_words,cur_tgt):
                # skip over special token(s)
                if txt_encoding['special_tokens_mask'][idx] == 1:
                    idx += [jj for jj,val in enumerate(txt_encoding['special_tokens_mask'][idx:]) if val == 0][0]

                word_tokens = tokenizer.tokenize(cur_word)
                new_tgt[idx:idx+len(word_tokens)] = [cur_label]*len(word_tokens)
                idx += len(word_tokens)
            
            cur_tgt = new_tgt.copy()                
        
        # check for defective observation (i.e. must have at least beginning and ending
        # "special" tokens for a valid observation (e.g. [CLS] tok1 tok2 ... [SEP] for
        # a BERT model)
        if sum(txt_encoding['special_tokens_mask']) >= 2:
            token_strings[obs_idx] = BertCommon['text_delimiter'].join(tokenized_text)                    
            
            # position
            tokenized_position = [None] * num_tokens
            for jj in range(num_tokens):
                tokenized_position[jj] = BertCommon['reserved_names']['position_prefix']+str(jj)

            position_strings[obs_idx] = BertCommon['text_delimiter'].join(tokenized_position)
            
            # segment
            if segment_vocab_size > 0:
                tokenized_segment = [None] * num_tokens
                for jj in range(num_tokens):
                    tokenized_segment[jj] = BertCommon['reserved_names']['segment_prefix']+str(seg_idx[jj])
                    
                segment_strings[obs_idx] = BertCommon['text_delimiter'].join(tokenized_segment)

            # target
            if cur_tgt is not None:
                if classification_problem:
                    if isinstance(cur_tgt, list):
                        # zero pad target list
                        target_array[obs_idx] = [str(0)]*max_seq_len
                        for jj, tgt in enumerate(cur_tgt):
                            target_array[obs_idx][jj] = str(int(tgt))
                            
                        tgtlen_array[obs_idx] = len(cur_tgt)
                    else:
                        target_array[obs_idx] = str(int(cur_tgt))
                        tgtlen_array[obs_idx] = 1
                else:
                    if isinstance(cur_tgt, list):
                        raise DLPyError('Multiple regression problems not supported.')
                    else:
                        target_array[obs_idx] = cur_tgt
                        tgtlen_array[obs_idx] = 1

            # weight
            if cur_wgt is not None:
                weight_array[obs_idx] = cur_wgt
                
            # extra variable(s)
            if cur_extra_var is not None:
                extra_var_array[obs_idx] = cur_extra_var
                
            # increment the valid observation index
            obs_idx += 1
        else:
            print('WARNING: observation #: ' + str(ii))
            raise DLPyError('Input string could not be tokenized.')
            
    if verbose:
        print("NOTE: all observations tokenized.\n")
            

    # reduce lists and inform user if one or more observations discarded
    if obs_idx < len(input_a):
        token_strings = token_strings[0:obs_idx]
        position_strings = position_strings[0:obs_idx]
        if segment_vocab_size > 0:
            segment_strings = segment_strings[0:obs_idx]
        if target is not None:
            target_array = target_array[0:obs_idx]
            tgtlen_array = tgtlen_array[0:obs_idx]
        if obs_weight is not None:
            weight_array = weight_array[0:obs_idx]
        if extra_var is not None:
            extra_var_array = extra_var_array[0:obs_idx]
            
        print('NOTE: observations with empty/invalid input or targets were discarded.  There are\n' 
              '' + str(obs_idx) + ' out of ' + str(len(input_a)) + ' observations remaining.\n')

    # inform user if one or more observations truncated
    if num_truncated > 0:
        print('WARNING: ' + str(num_truncated) + ' out of ' + str(len(input_a)) + ' observations exceeded the maximum sequence length\n'
              'These observations have been truncated so that only the first ' + str(max_seq_len) + ' tokens are used.\n')
              
    # set up variable names/types
    if segment_vocab_size > 0:
        var_names = [ds_vars['token_var'],ds_vars['position_var'],ds_vars['segment_var']]
        var_type = ['VARCHAR', 'VARCHAR', 'VARCHAR']
    else:
        var_names = [ds_vars['token_var'],ds_vars['position_var']]
        var_type = ['VARCHAR', 'VARCHAR']
    
    num_target_var = None
    if target is not None:
        if multiple_targets:
            num_target_var = max_seq_len
        else:
            num_target_var = 1
            
        var_names += generate_target_var_names(ds_vars,num_target_var)
        var_names += [ds_vars['target_len_var']]
        if classification_problem:
            var_type += ['VARCHAR']*num_target_var + ['NUMERIC']
        else:
            var_type += ['NUMERIC']*num_target_var + ['NUMERIC']
                
    if obs_weight is not None:
        var_names += [ds_vars['weight_var']]
        var_type += ['NUMERIC']
        
    if extra_var is not None:
        var_names += extra_var_names
        var_type += extra_var_types

    # check whether splitting to training/testing data sets or just a single data set
    if (train_fraction is not None) and (train_fraction > 0.0):
        np.random.seed(seed=seed)
        idx_prob = np.random.uniform(low=0.0, high=1.0,size=(obs_idx,))
        num_train = 0
        num_test = 0
        for ii in range(obs_idx):
            if idx_prob[ii] < train_fraction:
                num_train += 1
            else:
                num_test += 1
                
        # split data to train/test data sets
        # token, position, segment
        train_token_strings = [None] * num_train
        train_position_strings = [None] * num_train
        if segment_vocab_size > 0:
            train_segment_strings = [None] * num_train
        #
        test_token_strings = [None] * num_test
        test_position_strings = [None] * num_test
        if segment_vocab_size > 0:
            test_segment_strings = [None] * num_test
        # target
        if target is not None:
            train_target_array = [None] * num_train
            train_tgtlen_array = [None] * num_train
            #
            test_target_array = [None] * num_test
            test_tgtlen_array = [None] * num_test
        # weight 
        if obs_weight is not None:
            train_weight_array = [None] * num_train
            #
            test_weight_array = [None] * num_test
        # extra variable(s)
        if extra_var is not None:
            train_extra_var_array = [None] * num_train
            #
            test_extra_var_array = [None] * num_test
            
        train_idx = 0
        test_idx = 0
        for ii in range(obs_idx):
            if idx_prob[ii] < train_fraction:   # train data set
                train_token_strings[train_idx] = token_strings[ii]
                train_position_strings[train_idx] = position_strings[ii]
                if segment_vocab_size > 0:
                    train_segment_strings[train_idx] = segment_strings[ii]
                
                # NOTE: each element of train target array may be a value or a list
                if target is not None:
                    train_target_array[train_idx] = target_array[ii]
                    train_tgtlen_array[train_idx] = tgtlen_array[ii]
                    
                if obs_weight is not None:
                    train_weight_array[train_idx] = weight_array[ii]

                # NOTE: each element of train extra var array is a list
                if extra_var is not None:
                    train_extra_var_array[train_idx] = extra_var_array[ii]
                
                train_idx += 1
            else:                               # test data set
                test_token_strings[test_idx] = token_strings[ii]
                test_position_strings[test_idx] = position_strings[ii]
                if segment_vocab_size > 0:
                    test_segment_strings[test_idx] = segment_strings[ii]
                
                # NOTE: each element of test target array may be a value or a list
                if target is not None:
                    test_target_array[test_idx] = target_array[ii]                    
                    test_tgtlen_array[test_idx] = tgtlen_array[ii]
                    
                if obs_weight is not None:
                    test_weight_array[test_idx] = weight_array[ii]

                # NOTE: each element of test extra var array is a list
                if extra_var is not None:
                    test_extra_var_array[test_idx] = extra_var_array[ii]
                
                test_idx += 1
                
        # create CAS table for training data
        train_data_set = 'bert_train_data'
        if segment_vocab_size > 0:
            dlist = [train_token_strings, train_position_strings, train_segment_strings]
        else:
            dlist = [train_token_strings, train_position_strings]
            
        if target is not None:
            if isinstance(train_target_array[0],list):
                for ii in range(len(train_target_array[0])):
                    tmp_array = [train_target_array[jj][ii] for jj in range(train_idx)]
                    dlist += [tmp_array]
            else:   
                dlist += [train_target_array]
                
            dlist += [train_tgtlen_array]
            
        if obs_weight is not None:
            dlist += [train_weight_array]

        if extra_var is not None:
            for ii in range(len(train_extra_var_array[0])):
                tmp_array = [train_extra_var_array[jj][ii] for jj in range(train_idx)]
                dlist += [tmp_array]
        
        if verbose:
            print("NOTE: uploading training data to table " + train_data_set + ".")
            print("NOTE: there are " + str(num_train) + " observations in the training data set.\n")
            
        handler1 = BertDMH(dlist,var_names,var_type)
        conn.retrieve('table.addtable', _messagelevel='error',
                      table=train_data_set,
                      replace=True,
                      **handler1.args.addtable)
                           
        # create CAS table for test data
        test_data_set = 'bert_test_validation_data'
        if segment_vocab_size > 0:
            dlist = [test_token_strings, test_position_strings, test_segment_strings]
        else:
            dlist = [test_token_strings, test_position_strings]
        
        if target is not None:
            if isinstance(test_target_array[0],list):
                for ii in range(len(test_target_array[0])):
                    tmp_array = [test_target_array[jj][ii] for jj in range(test_idx)]
                    dlist += [tmp_array]
            else:
                dlist += [test_target_array]
                
            dlist += [test_tgtlen_array]
            
        if obs_weight is not None:
            dlist += [test_weight_array]

        if extra_var is not None:
            for ii in range(len(test_extra_var_array[0])):
                tmp_array = [test_extra_var_array[jj][ii] for jj in range(test_idx)]
                dlist += [tmp_array]

        if verbose:
            print("NOTE: uploading test/validation data to table " + test_data_set + ".")
            print("NOTE: there are " + str(num_test) + " observations in the test/validation data set.\n")
        
        handler2 = BertDMH(dlist,var_names,var_type)
        conn.retrieve('table.addtable', _messagelevel='error',
                      table=test_data_set,
                      replace=True,
                      **handler2.args.addtable)

        if verbose:
            print("NOTE: training and test/validation data sets ready.\n")
                           
        return num_target_var, train_data_set, test_data_set
    else:
        # single data set
        unified_data_set = 'bert_data'
        if segment_vocab_size > 0:
            dlist = [token_strings, position_strings, segment_strings]
        else:
            dlist = [token_strings, position_strings]
        
        if target is not None:
            if isinstance(target_array[0],list):
                for ii in range(len(target_array[0])):
                    tmp_array = [target_array[jj][ii] for jj in range(obs_idx)]
                    dlist += [tmp_array]
            else:
                dlist += [target_array]
                
            dlist += [tgtlen_array]
        
        if obs_weight is not None:
            dlist += [weight_array]

        if extra_var is not None:
            for ii in range(len(extra_var_array[0])):
                tmp_array = [extra_var_array[jj][ii] for jj in range(obs_idx)]
                dlist += [tmp_array]

        if verbose:
            print("NOTE: uploading data to table " + unified_data_set + ".")
            print("NOTE: there are " + str(obs_idx) + " observations in the data set.\n")
        
        handler = BertDMH(dlist,var_names,var_type)
        conn.retrieve('table.addtable', _messagelevel='error',
                      table=unified_data_set,
                      replace=True,
                      **handler.args.addtable)

        if verbose:
            print("NOTE: data set ready.\n")
                           
        return num_target_var, unified_data_set

def display_obs(conn, table_name, num_obs=5, random_draw=True, columns=None):
    '''
    Display observations from a given CAS table

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection.
    table_name : string
        Specifies the name of the CAS table.
    num_obs: int, optional
        Specifies the number of observations to display.
        Default: 5
    random_draw: boolean, optional
        Specifies whether observations are randomly chosen or not.
        Default: True
    columns : list of strings, optional
        Specifies the columns to display. If not specified, 
        a randomly selected column is displayed for each
        observation.
        Default: None
        
    '''    

    r = conn.retrieve('table.recordcount', _messagelevel='error',
                      table=table_name)
    num_rows = r['RecordCount']['N'].values[0]
    if random_draw:
        obs_index = np.random.randint(1,num_rows+1,min([num_rows,num_obs]))
    else:
        obs_index = [ii+1 for ii in range(min([num_rows,num_obs]))]
        
    for ii in obs_index:
        tmp = conn.retrieve('table.fetch', _messagelevel='error',
                            table=table_name, maxrows=1, from_=ii)
        col_names = list(tmp['Fetch'])
        
        print('------- Observation: ',ii,'-------\n')
        
        if columns is None:
            rint = random.randint(0,len(col_names)-1)
            print(col_names[rint],': ',tmp['Fetch'][col_names[rint]].to_list()[0])
            print('\n')
        else:
            for name in columns:
                if name in col_names:
                    print(name,': ',tmp['Fetch'][name].to_list()[0])
                else:
                    print('Column ' + name + ' not found in ' + table_name + '.')
                print('\n')

def bert_summary(conn, table_name, full_table=True, subset_fraction=0.1):
    '''
    Display summary statistics for tokenized data from a given CAS table

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection.
    table_name : string
        Specifies the name of the CAS table.
    full_table: boolean, optional
        Specifies whether statistics are calculated over full table or subset.
        Default: True
    subset_fraction : float, optional
        Specifies the fraction of the table to use to calculate summary
        statistics.  May be necessary for large tables.
        Default: 0.1
        
    '''    

    r = conn.retrieve('table.recordcount', _messagelevel='error',
                      table=table_name)
    num_obs = r['RecordCount']['N'].values[0]
    print("NOTE: there are " + str(num_obs) + " observations in the Viya table.")

    if not full_table:
        num_obs_calc = int(round(num_obs*subset_fraction))
        print("NOTE: calculating summary statistics based on the first " + str(round(subset_fraction*100.0)) + "% of the table.\n")
    else:
        num_obs_calc = num_obs
    
    chunk_size = min([num_obs_calc, 10000])
    min_tokens = sys.maxsize
    max_tokens = 0
    sum_tokens = 0
    sum_sq_tokens = 0
    token_var = BertCommon['variable_names']['token_var']
    for ii in range(0,num_obs_calc,chunk_size):
        num_rows = min([chunk_size,num_obs_calc-ii])
        tmp = conn.retrieve('table.fetch', _messagelevel='error',
                            table=table_name, maxrows=num_rows, from_=ii, to=ii+num_rows)
                            
        col_names = list(tmp['Fetch'])
        if token_var not in list(tmp['Fetch']):
            raise DLPyError("Missing variable " + token_var + " in table " + table_name + ".")
        
        tmp_list = tmp['Fetch'][token_var].to_list()
        obs_num_tokens = [len(tmp_list[jj].split(' ')) for jj in range(num_rows)]
        min_tokens = min([min_tokens, min(obs_num_tokens)])
        max_tokens = max([max_tokens, max(obs_num_tokens)])
        sum_tokens += sum(obs_num_tokens)
        sum_sq_tokens += sum([x1*x2 for x1,x2 in zip(obs_num_tokens, obs_num_tokens)])
        
    mean_num_tokens = sum_tokens/num_obs_calc
    std_num_tokens = np.sqrt(sum_sq_tokens/num_obs_calc - mean_num_tokens*mean_num_tokens)
        
    print("NOTE: minimum number of tokens in an observation = " + str(min_tokens))
    print("NOTE: maximum number of tokens in an observation = " + str(max_tokens))
    print("NOTE: average number of tokens in an observation = " + str(mean_num_tokens))
    print("NOTE: standard deviation of the number of tokens in an observation = " + str(std_num_tokens) + '\n')
