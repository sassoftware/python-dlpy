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

''' Extended attribute table functions for the DLPy package '''

import os
import platform
import numpy as np
import pandas as pd
import swat as sw
import string
import warnings
import struct
from dlpy.model import DataSpec
from dlpy.layers import Layer
from dlpy.utils import DLPyError

def create_extended_attributes(conn, model_name, layers, data_spec, label_file_name=None):

    '''
    Create/override extended model attributes for given model

    Update the extended model attributes given data spec(s).
    The data spec(s) must define all relevant dictionary elements for
    class :class:`DataSpec` or the resulting attribute table will
    be inaccurate.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    layers : list of :class:`Layer`
        Specifies all the layers in the deep learning model
    data_spec: list of :class:`DataSpec`
        data specification for input and output layer(s)
    label_file_name: string, optional
        Fully qualified path to CSV file containing user-defined 
        classification labels.  If not specified, numeric labels
        are used.
        
    '''

    # ensure list of layers
    if not isinstance(layers,list):
        raise TypeError('Parameter layers must be a list of Layer objects.')
    else:
        if not all(isinstance(x,Layer) for x in layers):
            raise TypeError('Some elements of the layers list are not Layer objects.')

    # ensure list of data specs
    if not isinstance(data_spec,list):
        raise TypeError('Parameter data_spec must be a list of DataSpec objects.')
    else:
        if not all(isinstance(x,DataSpec) for x in data_spec):
            raise TypeError('Some elements of the data_spec list are not DataSpec objects.')

    # read user-supplied labels
    if label_file_name is None:
        labels = None
    else:
        all_label_info = pd.read_csv(label_file_name, skipinitialspace=True, index_col=False)
        labels = all_label_info['label'].values.tolist()
    
    # ensure table action loaded
    rt = conn.queryactionset('table', _messagelevel = 'error')
    if not rt:
        conn.loadactionset('table', _messagelevel = 'error')
        
    # parse data spec(s) and create data spec attributes
    ds_info = create_dataspec_attributes(conn, model_name, layers, data_spec)
    
    # update variable list attributes
    create_varlist_attributes(conn, model_name, layers, ds_info)

    # update variable information attributes
    create_varinfo_attributes(conn, model_name, layers, ds_info, labels)

    # update input parameter attributes
    create_inputparm_attributes(conn, model_name, layers, ds_info)

def create_dataspec_attributes(conn, model_name, layers, data_spec):

    '''
    Create/override extended model attributes for data spec(s)

    Update the extended model attributes for the data spec(s).
    The data spec must define all relevant dictionary elements for
    class :class:`DataSpec` or the resulting attribute table will
    be inaccurate.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    layers : list of :class:`Layer`
        Specifies all the layers in the deep learning model
    data_spec: list of :class:`DataSpec`
        data specification for input and output layer(s)

    Returns
    -------
    dict
        data spec and variable information
        
    '''
                   
    # collect dataspec(s) associated with input and task layers
    input_data_spec = []
    task_data_spec = []
    ds_layer_type = []
    for spec in data_spec:
        for layer in layers:
            if spec['layer'] == layer.name:
                if layer.type == 'input':
                    input_data_spec.append(spec)
                    ds_layer_type.append('input')
                else:
                    task_data_spec.append(spec)
                    ds_layer_type.append('task')
                    
    sorted_data_spec = input_data_spec + task_data_spec
                        
    # character/string attributes
    layer_names = bytearray()
    var_names = bytearray()
    len_var_names = bytearray()
    
    # int64_t attributes
    data_type = bytearray()
    token_size = bytearray()
    
    # int attributes
    n_vars = bytearray()
    n_nominal_vars = bytearray()
    layer_name_lens = bytearray()
    var_names_lens = bytearray()
    len_name_lens = bytearray()  

    # double attributes
    loss_scale_factor = bytearray()

    # information pertaining to all variables included in all data specs 
    all_vars = {'name':[], 
                'ds_type':[],
                'nominal':[],
                'rtype':[],
                'rawlen':[],
                'fmt_name':[],
                'fmt_nfl':[],
                'fmt_nfd':[],
                'fmt_datalen':[],
                'levels':[]}
    nom_vars = {'name':[]}
    
    # input layer(s) followed by task layer(s)
    n_input_vars = 0
    for ii in range(len(sorted_data_spec)): 
        spec = sorted_data_spec[ii]
        layer_type = ds_layer_type[ii]
        
        # loss scale factor
        loss_scale_factor = loss_scale_factor + struct.pack('@d',1.0)
        
        # data type (int64)
        data_info = sas_var_info(spec.type)
        data_type = data_type + struct.pack('@q',data_info['ds_type'])

        # token size (int64)
        if "numeric_nominal_parms" in spec:
            if "token_size" in spec["numeric_nominal_parms"]:
                token_size = token_size + \
                             struct.pack('@q',spec["numeric_nominal_parms"]["token_size"])
            else:
                token_size = token_size + \
                             struct.pack('@q',0)
        else:
            token_size = token_size + \
                         struct.pack('@q',0)
        
        # number of variables
        n_vars = n_vars + struct.pack('@i',len(spec['data']))

        # number of nominal variables
        if "nominals" in spec:
            n_nominal_vars = n_nominal_vars + struct.pack('@i',len(spec['nominals']))
        else:
            n_nominal_vars = n_nominal_vars + struct.pack('@i',0)
        
        # layer names
        barray = bytearray(spec['layer'],encoding = 'utf-8')
        layer_names = layer_names + barray
        layer_name_lens = layer_name_lens + struct.pack('@i',len(barray))
        
        # variable names and lengths (only first variable for dataspec)
        barray = bytearray(spec['data'][0],encoding = 'utf-8')
        var_names = var_names + barray
        var_names_lens = var_names_lens + struct.pack('@i',len(barray))

        # collect information for all variables
        for var in spec['data']:
            all_vars['name'].append(var.encode('utf-8'))
            all_vars['ds_type'].append(data_info['ds_type'])
            all_vars['nominal'].append(False)
            all_vars['rtype'].append(data_info['rtype'])
            all_vars['rawlen'].append(data_info['rawlen'])
            all_vars['fmt_name'].append(data_info['fmt_name'])
            all_vars['fmt_nfl'].append(data_info['fmt_nfl'])
            all_vars['fmt_nfd'].append(data_info['fmt_nfd'])
            all_vars['fmt_datalen'].append(data_info['fmt_datalen'])
            all_vars['levels'].append(0)
            
        # update the number of input variables
        if layer_type == 'input':
            n_input_vars = n_input_vars + len(spec['data'])
            
        # all nominal variable names
        if "nominals" in spec:
            for var in spec['nominals']:
                try:
                    index = all_vars['name'].index(var.encode('utf-8'))
                    all_vars['nominal'][index] = True
                    nom_vars['name'].append(var.encode('utf-8'))
                except ValueError:
                    raise DLPyError('You specified a nominal variable that does\n'
                                    'not exist in the variable list.')
                    
        # length variable names (RNN only)
        if "numeric_nominal_parms" in spec:
            if "length" in spec["numeric_nominal_parms"]:
                barray = bytearray(spec["numeric_nominal_parms"]["length"], 
                                   encoding = 'utf-8')
                len_var_names = len_var_names + barray
                len_name_lens = len_name_lens + struct.pack('@i',len(barray))
                # add to all variable information
                # NOTE: length variable is numeric/nominal type
                numnom_info = sas_var_info('NUMNOM')
                all_vars['name'].append(spec["numeric_nominal_parms"]["length"].encode('utf-8'))
                all_vars['ds_type'].append(numnom_info['ds_type'])
                all_vars['nominal'].append(False)
                all_vars['rtype'].append(numnom_info['rtype'])
                all_vars['rawlen'].append(numnom_info['rawlen'])
                all_vars['fmt_name'].append(numnom_info['fmt_name'])
                all_vars['fmt_nfl'].append(numnom_info['fmt_nfl'])
                all_vars['fmt_nfd'].append(numnom_info['fmt_nfd'])
                all_vars['fmt_datalen'].append(numnom_info['fmt_datalen'])
                all_vars['levels'].append(0)
                # update the number of input variables
                if layer_type == 'input':
                    n_input_vars = n_input_vars + 1
                
            else:
                barray = bytearray(" ", encoding = 'utf-8')
                len_var_names = len_var_names + barray
                len_name_lens = len_name_lens + struct.pack('@i',0)
        else:
            barray = bytearray(" ", encoding = 'utf-8')
            len_var_names = len_var_names + barray
            len_name_lens = len_name_lens + struct.pack('@i',0)
        
    # update parameters for attribute set dl_dataspecs_parms
    set_name = "dl_dataspecs_parms".encode('utf-8')
    
    # number of data specs
    update_attr(conn, model_name, [len(data_spec)], set_name, "nDataSpecs", "int")

    # data spec data types
    update_attr(conn, model_name, data_type, set_name, "dataTypes", "int64")

    # token sizes
    update_attr(conn, model_name, token_size, set_name, "tokenSizes", "int64")

    # number of variables
    update_attr(conn, model_name, n_vars, set_name, "nVars", "int")

    # number of nominal variables
    update_attr(conn, model_name, n_nominal_vars, set_name, "nNominalVars", "int")

    # layer names
    update_attr(conn, model_name, layer_names.decode('utf-8'), set_name, "layerNames", "char")

    # layer name lengths
    update_attr(conn, model_name, layer_name_lens, set_name, "layerNameLens", "int")

    # data spec variable names
    update_attr(conn, model_name, var_names, set_name, "varNames", "binary")

    # data spec variable name lengths
    update_attr(conn, model_name, var_names_lens, set_name, "varNamesLens", "int")

    # data spec length variable names
    update_attr(conn, model_name, len_var_names.decode('utf-8'), set_name, "lenVarNames", "char")
    
    # data spec length variable name lengths
    update_attr(conn, model_name, len_name_lens, set_name, "lenNameLens", "int")

    # loss scale factor
    update_attr(conn, model_name, loss_scale_factor, set_name, "lossScaleFactor", "double")
    
    # create dictionary needed by other attribute functions
    ds_dict = {"all_vars" : all_vars,
               "nom_vars" : nom_vars,
               "spec_list" : sorted_data_spec,
               "n_input_vars" : n_input_vars}
                              
    return ds_dict
                                  
def create_varlist_attributes(conn, model_name, layers, ds_info):

    '''
    Create/override extended model attributes for variable(s)

    Update the extended model attributes for the model variable(s).
    The data spec attribute(s) must have been created prior to
    calling this function.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    layers : list of :class:`Layer`
        Specifies all the layers in the deep learning model
    ds_info: dictionary
        parsed data spec information

    '''
    
    # update parameters for attribute set dl_dataspecs_parms
    set_name = "dl_model_varlist".encode('utf-8')
    
    # number of model variables
    update_attr(conn, model_name, [len(ds_info["all_vars"]["name"])], set_name, "var_ntot", "int")
    
    # generate variable list attributes
    var_rtype = bytearray()
    var_rawlen = bytearray()
    var_list = bytearray()
    null_byte = bytearray('\u0000',encoding='utf-8')
    for ii in range(len(ds_info['all_vars']['name'])):
        barray = bytearray(ds_info['all_vars']['name'][ii])
        var_list = null_byte.join([var_list, barray])
        var_rtype = var_rtype + struct.pack('@i',ds_info['all_vars']['rtype'][ii])
        var_rawlen = var_rawlen + struct.pack('@i',ds_info['all_vars']['rawlen'][ii])
        
    # finalize variable list
    var_list = var_list[1:] + null_byte
 
    # update parameters for attribute set dl_dataspecs_parms
    set_name = "dl_model_varlist".encode('utf-8')
     
    # variable list
    update_attr(conn, model_name, var_list, set_name, "var_list", "binary")

    # variable root type
    update_attr(conn, model_name, var_rtype, set_name, "var_rtype", "int")

    # variable root type
    update_attr(conn, model_name, var_rawlen, set_name, "var_rawlen", "int")
    
def create_varinfo_attributes(conn, model_name, layers, ds_info, labels=None):

    '''
    Create/override extended model attributes for variable information

    Update the extended model attributes for the variable information.
    The data spec attribute(s) must have been created prior to 
    calling this function.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    layers : list of :class:`Layer`
        Specifies all the layers in the deep learning model
    ds_info: dictionary
        parsed data spec information
    labels: list, optional
        list of string values representing class labels
    '''

    # update parameters for attribute set dl_dataspecs_parms
    set_name = "dl_model_varinfo".encode('utf-8')
    all_vars = ds_info['all_vars']
    
    # format information
    fmt_name = bytearray()
    fmt_namelen = bytearray()
    fmt_nfl = bytearray()
    fmt_nfd = bytearray()  
    fmt_datalen = bytearray()
    
    # set format attributes for all variables
    null_byte = bytearray('\u0000',encoding='utf-8')
    for ii in range(len(ds_info['all_vars']['name'])):
        tmp_name = ds_info['all_vars']['fmt_name'][ii].encode('utf-8')
        barray = bytearray(tmp_name)
        fmt_name = null_byte.join([fmt_name, barray])
        fmt_namelen = fmt_namelen + struct.pack('@i',len(tmp_name))
        fmt_nfl = fmt_nfl + struct.pack('@i',ds_info['all_vars']['fmt_nfl'][ii])
        fmt_nfd = fmt_nfd + struct.pack('@i',ds_info['all_vars']['fmt_nfd'][ii])
        fmt_datalen = fmt_datalen + struct.pack('@i',ds_info['all_vars']['fmt_datalen'][ii])
    
    # finalize format name list
    fmt_name = fmt_name[1:] + null_byte
    
    # format names
    update_attr(conn, model_name, fmt_name, set_name, "fmt_name", "binary")

    # format name length
    update_attr(conn, model_name, fmt_namelen, set_name, "fmt_namelen", "binary")
    
    # format nfl
    update_attr(conn, model_name, fmt_nfl, set_name, "fmt_nfl", "binary")
    
    # format nfd
    update_attr(conn, model_name, fmt_nfd, set_name, "fmt_nfd", "binary")
    
    # format data length
    update_attr(conn, model_name, fmt_datalen, set_name, "fmt_datalen", "binary")

    # nominal variable level information
    level_name = bytearray()
    level_namelen = bytearray()
    
    # set level information for nominal variables
    for spec in ds_info['spec_list']:
        # determine layer type
        for layer in layers:
            if spec['layer'] == layer.name:
                break
    
        if "nominals" in spec:
            n_nom_var = len(spec['nominals'])
            if n_nom_var > 0:
                if layer.type == 'input':
                    raise DLPyError('Setting attributes for non-numeric input layer variables\n'
                                    'is not supported.\n')
                                    
                elif layer.type == 'output':
                    if layer.config['n'] is None:
                        raise DLPyError('You must specify the number of neurons for the output\n'
                                        'layer variables when setting attributes.\n')
                                        
                    n_levels = int(layer.config['n'])
                    task_type = '0x8'
                    
                    # create needed labels for nominal variables
                    ljust_labels = create_class_labels(n_levels, labels)

                elif layer.type == 'detection':
                    if layer.config['class_number'] is None:
                        raise DLPyError('You must specify the number of classes for the object\n'
                                        'detection when setting attributes.\n')
                                        
                    
                    n_levels = int(layer.config['class_number'])
                    task_type = '0x800000'
                    
                    # create needed labels for nominal variables
                    ljust_labels = create_class_labels(n_levels, labels)
                                    
                else:
                    raise DLPyError('Attributes can only be set for variables defined in input,\n'
                                    'output, or detection layers defined in data specifications.\n')
                    
                # create level names for all nominal variables and all levels
                for ii in range(n_nom_var):
                    nom_name = spec['nominals'][ii].encode('utf-8')
                    index = all_vars['name'].index(nom_name)
                    all_vars['levels'][index] = n_levels
                    for jj in range(n_levels):
                        level_name = level_name + bytearray(ljust_labels[jj].encode('utf-8'))
                        level_namelen = level_namelen + struct.pack('@i',len(ljust_labels[jj]))
                        
        else:
            task_type = '0x10'
                
    # update level names/lengths if any nominal variables
    if len(level_name):
        # level name
        update_attr(conn, model_name, level_name, set_name, "level_name", "binary")
    
        # level name length
        update_attr(conn, model_name, level_namelen, set_name, "level_namelen", "int")
        
    # number of levels for all variables 
    levels = bytearray()
    for lval in all_vars['levels']:
        levels = levels + struct.pack('@q',lval)

    # levels
    update_attr(conn, model_name, levels, set_name, "level_info", "int64")

    # model_task
    update_attr(conn, model_name, [int(task_type,16)], set_name, "model_task", "int")
    
def create_inputparm_attributes(conn, model_name, layers, ds_info):

    '''
    Create/override extended model attributes for input parameters

    Update the extended model attributes for the input parameters.
    The data spec attribute(s) must have been created prior to 
    calling this function.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    layers : list of :class:`Layer`
        Specifies all the layers in the deep learning model
    ds_info: dictionary
        parsed data spec information

    '''
        
    # update parameters for attribute set dl_dataspecs_parms
    set_name = "dl_input_parms".encode('utf-8')
        
    # generate target variable list attributes
    target_var_list = bytearray()
    null_byte = bytearray('\u0000',encoding='utf-8')
    for ii in range(ds_info['n_input_vars'],len(ds_info['all_vars']['name'])):
        barray = bytearray(ds_info['all_vars']['name'][ii])
        target_var_list = null_byte.join([target_var_list, barray])

    # finalize target variable list
    target_var_list = target_var_list[1:] + null_byte
    
    # target variable list
    update_attr(conn, model_name, target_var_list, set_name, "target", "binary")
    
    # generate nominal variable list attributes
    if len(ds_info['nom_vars']) > 0:
        nominal_var_list = bytearray()
        for ii in range(len(ds_info['nom_vars']['name'])):
            barray = bytearray(ds_info['nom_vars']['name'][ii])
            nominal_var_list = null_byte.join([nominal_var_list, barray])

        # finalize nominal variable list
        nominal_var_list = nominal_var_list[1:] + null_byte
    
        # update nominal variable list
        update_attr(conn, model_name, nominal_var_list, set_name, "nominal", "binary")

    else:
        # no nominal variables - drop nominal attribute
        rt = conn.retrieve('table.attribute',
                           _messagelevel = 'error',
                           name=model_name + '_weights',
                           attributes=[{"key":"nominal"}],
                           set=set_name,
                           task="DROP")
                           
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('Cannot drop attribute, there seems to be a problem.')
    
def sas_var_info(var_type):

    '''
    Returns SAS variable type information

    Extracts variable information needed to update extended
    attribute table.

    Parameters
    ----------
    var_type : string
        Specifies the type of the input data in the data spec.
        Valid Values: NUMERICNOMINAL, NUMNOM, TEXT, IMAGE, OBJECTDETECTION

    Returns
    -------
    dict
        SAS variable information
        
    '''

    if var_type.lower() in ["numericnominal", "numnom"]:
        var_info = {"ds_type" : 1,
                    "rtype" : 1,
                    "rawlen" : 8,
                    "fmt_name" : "BEST",
                    "fmt_nfl" : 12,
                    "fmt_nfd" : 0,
                    "fmt_datalen" : 12}
    elif var_type.lower() == "text":
        raise DLPyError('Attribute updating not supported for text variable(s).')
    elif var_type.lower() == "image":
        var_info = {"ds_type" : 3,
                    "rtype" : 0,
                    "rawlen" : 1000000,
                    "fmt_name" : "BEST",
                    "fmt_nfl" : 0,
                    "fmt_nfd" : 0,
                    "fmt_datalen" : 1}
    elif var_type.lower() == "objectdetection":
        var_info = {"ds_type" : 4,
                    "rtype" : 1,
                    "rawlen" : 8,
                    "fmt_name" : "BEST",
                    "fmt_nfl" : 12,
                    "fmt_nfd" : 0,
                    "fmt_datalen" : 12}
    else:
        raise DLPyError('The variable type is invalid. Only NUMERICNOMINAL,\n'
                        'NUMNOM, TEXT, IMAGE, and OBJECTDETECTION are supported.')
        
    return var_info

def update_attr(conn, model_name, attr_value, attr_set, attr_key, attr_type):
 
    '''
    Update individual extended model attributes

    Key/value pair required to specify extended attributes.  Provide
    correct syntax for calling attribute action.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    attr_value : list of bytes, int, int64, double, or char
        Numeric/character representation of attribute
    attr_set : string
        Name of attribute set to update
    attr_key : string
        Key name of attribute
    attr_type : string
        One of double, int64, int, char, or binary

    '''
 
    if attr_type.lower() in ['double', 'int64', 'int', 'char', 'binary']:
        if attr_type.lower() == 'char':
            attr_helper(conn, model_name, attr_set, attr_key, attr_value)
        else:
            if len(attr_value) > 1:
                      
                # create binary blob using SWAT 
                attr_blob = sw.blob(attr_value)
            
                # write attribute
                attr_helper(conn, model_name, attr_set, attr_key, attr_blob)
            
            else:
                attr_helper(conn, model_name, attr_set, attr_key, attr_value[0])
    else:
        raise TypeError('Extended table attributes must be one of :\n'
                        '1. character string;\n'
                        '2. double precision value/list,\n'
                        '3. int64 value/list,\n'
                        '4. int value/list,\n'
                        '5. binary blob.')
    
            
def attr_helper(conn, model_name, attr_set, attr_key, attr_blob):
            
    '''
    Call action to update individual extended model attribute

    Key/value pair required to specify extended attributes.  Provide
    correct syntax for calling attribute action.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    attr_set : string
        Name of attribute set to update
    attr_key : string
        Key name of attribute
    attr_blob : double, int64, int, char, or binary blob
        Representation of attribute

    '''
            
    # drop existing attribute
    rt = conn.retrieve('table.attribute',
                       _messagelevel = 'error',
                       name=model_name + '_weights',
                       attributes=[{"key":attr_key}],
                       set=attr_set,
                       task="DROP")
    
    # NOTE: ignore errors if attribute or attribute set
    # doesn't exist
        
    # add new attribute
    rt = conn.retrieve('table.attribute',
                       _messagelevel = 'error',
                       name=model_name + '_weights',
                       attributes=[{"key":attr_key,"value":attr_blob}],
                       set=attr_set,
                       task="ADD")
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError('Cannot add attribute, there seems to be a problem.')

def export_attr_xml(conn, model_name, file_name):

    '''
    Create XML version of extended attribute table

    Call action to create XML blob containing model attributes.
    Write resulting blob to text file.

    Parameters
    ----------
    conn : CAS
        The CAS connection object
    model_name : string
        Specifies the name of the deep learning model
    file_name : string
        Name of XML file

    '''

    rt = conn.retrieve('table.attribute',
                       _messagelevel = 'error',
                       name=model_name + '_weights',
                       task="EXPORT",
                       xml="attr")
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError('Cannot export model attributes, there seems to be a problem.')

    ascii_text = rt['xmlblob'].decode('utf8')
    with open(file_name, "w") as myfile:
        myfile.write(ascii_text)
    myfile.close()

def create_class_labels(n_levels, labels=None):

    '''
    Create class labels

    Create class labels with or without user-defined labels.

    Parameters
    ----------
    n_levels : integer
        The number of levels for each classification variable.
    labels : list of string or None
        Specifies the class labels

    Returns
    -------
    list
        Left-justified class labels.

    '''

    # create needed labels for nominal variables
    ljust_labels = []
    if labels is None:
        # strictly numeric labels (e.g. 0, 1, ...)
        for ii in range(n_levels):
            ljust_labels.append(str(ii).ljust(12))
    else:
        # user-supplied labels
        if n_levels != len(labels):
            raise DLPyError('The number of class labels does not match\n'
                            'the number of class levels for object detection.\n')
        else:
            for lval in labels:
                if len(lval) > 12:
                    ljust_labels.append(lval[:12])
                else:
                    ljust_labels.append(lval.ljust(12))

    return ljust_labels