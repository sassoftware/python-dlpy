#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

''' Supporting functions for keras model conversion '''

import os
import sys
import h5py
import numpy as np
from keras import backend as K
from dlpy.utils import DLPyError
from .model_conversion_utils import replace_forward_slash, remove_layer_wrapper, create_cpu_compatible_layer

rnn_cpu_layer_classes = ['simplernn', 'lstm', 'gru']
rnn_gpu_layer_classes = ['cudnnlstm', 'cudnngru']
rnn_layer_classes = rnn_cpu_layer_classes + rnn_gpu_layer_classes
conv_layer_classes = ['conv1d', 'conv2d', 'separableconv1d', 'separableconv2d', 'depthwiseconv2d', 'conv2dtranspose']

try:
    from keras.engine.topology import preprocess_weights_for_loading
except ImportError:
    from keras.engine.saving import preprocess_weights_for_loading

# let Keras read parameters and then transform to format needed for SAS deep learning
# NOTE: modified version of Keras function load_weights_from_hdf5_group()
def write_keras_hdf5_from_file(model, rnn_support, hdf5_in, hdf5_out):
    '''
    Generate an HDF5 file with trained model parameters given a Keras definition

    Parameters
    ----------
    model : Keras model
       Keras deep learning model
    rnn_support : boolean
       Indicates whether importing RNN models is supported
    hdf5_in : string
       Fully qualified file name of Keras HDF5 file
    hdf5_out : string
       Fully qualified file name of SAS-compatible HDF5 file

    '''
    # open input/output files
    if os.path.isfile(hdf5_in):
        f_in = h5py.File(hdf5_in, 'r')
        try:
            f_out = h5py.File(hdf5_out, 'w')
        except IOError:
            raise DLPyError('The specified file cannot be written: ' + hdf5_out)
    else:
        raise DLPyError('The specified file does not exist: ' + hdf5_in)

    if 'keras_version' in f_in.attrs:
        original_keras_version = f_in.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'

    if 'backend' in f_in.attrs:
        original_backend = f_in.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    model_type = None
    use_gpu = None
    try:
        # determine model type
        # NOTE: must check ALL RNN layers to determine
        #       whether model must run on GPU
        gpu_layers = []
        cpu_layers = []
        for layer in model.layers:
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                # check for RNN layers
                if class_name in rnn_layer_classes:
                    model_type = 'RNN'
                    image_data_format = None
                    if class_name in rnn_gpu_layer_classes:
                        gpu_layers.append(True)
                    elif class_name in rnn_cpu_layer_classes:
                        cpu_layers.append(True)

        # verify that model is supported by SAS Deep Learning
        if model_type == 'RNN':
            if rnn_support:
                if (len(gpu_layers) > 0) and (len(cpu_layers) == 0):
                    use_gpu = True
                elif (len(gpu_layers) == 0) and (len(cpu_layers) > 0):
                    use_gpu = False
                elif (len(gpu_layers) > 0) and (len(cpu_layers) > 0):
                    raise DLPyError('A mixture of CPU and GPU layers was detected. '
                                    'This is not supported by SAS Deep Learning.')
            else:
                raise DLPyError('RNN model detected: your Viya deployment does not support '
                                'importing an RNN model.')
                
        if model_type is None:
            found_cnn_layer = False
            for layer in model.layers:
                class_name, sublayers = remove_layer_wrapper(layer)
                for tlayer in sublayers:
                    # check for CNN layers
                    if class_name in conv_layer_classes:
                        model_type = 'CNN'
                        image_data_format = K.image_data_format()
                        found_cnn_layer = True
                
                if found_cnn_layer:
                    break

        if model_type is None:
            raise DLPyError('Only RNN and CNN models are currently supported.')
            
        # navigate to correct HDF5 group
        if 'layer_names' in f_in.attrs.keys():
            root_group = f_in
        elif 'layer_names' in f_in['model_weights'].attrs.keys():
            root_group = f_in['model_weights']
        else:
            raise DLPyError('Cannot read HDF5 file correctly')
        
        # determine layers with weights
        filtered_layers = []
        for layer in model.layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)

        layer_names = [n.decode('utf8') for n in root_group.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = root_group[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if weight_names:
                filtered_layer_names.append(name)
        
        layer_names = filtered_layer_names
        if len(layer_names) != len(filtered_layers):
            raise ValueError('You are trying to load a weight file '
                             'containing ' + str(len(layer_names)) +
                             ' layers into a model with ' +
                             str(len(filtered_layers)) + ' layers.')

        # determine permutation vector associated with flattening layer (if it exists)
        if model_type == 'CNN':
            flatten_layer_index = -1
            index = 0
            for layer in model.layers:
                if layer.__class__.__name__.lower() == 'flatten':
                    flatten_layer_index = index
                    break
                index = index + 1

            if flatten_layer_index != -1:
                layer = model.layers[flatten_layer_index]
                permute_layer_name = model.layers[flatten_layer_index + 1].name
                if image_data_format == 'channels_first':
                    C, H, W = (layer.input_shape)[1:]
                else:
                    H, W, C = (layer.input_shape)[1:]
                N = (layer.output_shape)[1]
                perm_index = [0] * N
                if image_data_format == 'channels_last':
                    ii = 0
                    for cc in range(C):
                        for hh in range(H):
                            for ww in range(W):
                                perm_index[ii] = hh * W * C + ww * C + cc
                                ii = ii + 1
                else:
                    for nn in range(N):
                        perm_index[nn] = nn
            else:
                perm_index = []
                permute_layer_name = None
        else:
            perm_index = []
            permute_layer_name = None

        # populate attributes with layer names
        attrib_layer_names = []
        for name in layer_names:
            layer = model.get_layer(name=name)
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                attrib_layer_names.append(tlayer.name)
                            
        f_out.attrs['layer_names'] = [replace_forward_slash(l).encode('utf8') for l in attrib_layer_names]
        # let Keras read weights, reformat, and write to SAS-compatible file
        for k, name in enumerate(layer_names):
            g_in = root_group[name]
            layer = filtered_layers[k]

            weight_names = [n.decode('utf8') for n in g_in.attrs['weight_names']]
            weight_values = [g_in[weight_name] for weight_name in weight_names]
            symbolic_weights = layer.weights
                        
            # create CPU-compatible layer
            cpu_layer = create_cpu_compatible_layer(layer, model_type)
                                        
            # use Keras to load/preprocess weights
            weight_values = preprocess_weights_for_loading(cpu_layer,
                                                           weight_values,
                                                           original_keras_version,
                                                           original_backend)
                                                                   
            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '" in the current model) was found to '
                                 'correspond to layer ' + name +
                                 ' in the saved file. '
                                 'However the new layer ' + layer.name +
                                 ' expects ' + str(len(symbolic_weights)) +
                                 ' weights, but the saved weights have ' +
                                 str(len(weight_values)) +
                                 ' elements.')
            if layer.__class__.__name__.lower() == 'batchnormalization':
                bn_gamma = np.ones(weight_values[0].shape,
                                   dtype=weight_values[0].dtype)
                bn_beta = np.zeros(weight_values[0].shape,
                                   dtype=weight_values[0].dtype)
                                   
                layer_config = layer.get_config()
                
                # if scale = False and center = True
                if not layer_config['scale'] and layer_config['center']:
                    weight_values.insert(0, bn_gamma)
                    weight_names.insert(0, replace_forward_slash(layer.name)+'/'+'gamma:0')
                # if scale = True and center = False
                elif layer_config['scale'] and not layer_config['center']:
                    weight_values.insert(1, bn_beta)
                    weight_names.insert(1, replace_forward_slash(layer.name)+'/'+'beta:0')
                # if scale = False and center = False
                elif not layer_config['scale'] and not layer_config['center']:
                    weight_values = [bn_gamma, bn_beta] + weight_values
                    weight_names = [replace_forward_slash(layer.name)+'/'+'gamma:0', 
                                    replace_forward_slash(layer.name)+'/'+'beta:0'] + weight_names
                                    
                # add epsilon to variance values to avoid divide by zero
                if 'epsilon' in layer_config.keys():
                    for ii,wgt_name in enumerate(weight_names):
                        if 'moving_variance' in wgt_name:
                            weight_values[ii] = weight_values[ii] + (layer_config['epsilon']*
                                                                     np.ones(weight_values[ii].shape,
                                                                             dtype=weight_values[ii].dtype))
                    
            # read/write weights
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                g_out = f_out.create_group(replace_forward_slash(tlayer.name))
                new_weight_names = []
                wgt_idx = 0
            
                for ii,wgt_name in enumerate(weight_names):
                    if tlayer.name in wgt_name:
                        if type(weight_values[ii]) == np.ndarray:
                            tensor_in = weight_values[ii]
                        else:
                            tensor_in = np.zeros(weight_values[ii].shape,
                                                dtype=weight_values[ii].dtype)
                            weight_values[ii].read_direct(tensor_in)

                        # permute axes as needed to conform to SAS deep
                        # learning "channels first" format
                        if (image_data_format is not None) and (image_data_format == 'channels_first'):
                            # format: (C,fdim1, fdim2, fdim3) ==> (C,fdim3,fdim1,fdim2)
                            if len(tensor_in.shape) == 4:
                                tensor_out = np.transpose(tensor_in, (0, 3, 1, 2))
                            else:
                                tensor_out = tensor_in.copy()
                        else:
                            # "channels last" format or not image processing problem
                            
                            # process RNN layers first
                            if class_name in rnn_layer_classes:
                                cpu_class_name, cpu_sublayers = remove_layer_wrapper(cpu_layer)
                                if (len(tensor_in.shape) == 1) and (class_name != cpu_class_name):
                                    tensor_out = np.tile(0.5 * tensor_in, 2)
                                else:
                                    tensor_out = tensor_in.copy()
                            # not an RNN layer, but this is a vector - nothing to permute
                            elif len(tensor_in.shape) == 1:
                                tensor_out = tensor_in.copy()
                            else:
                                # permute Conv2D tensor to "channels_first" format
                                if class_name == 'conv2d':
                                    tensor_out = np.transpose(tensor_in, (3, 2, 0, 1))
                                # have to account for neuron ordering in first dense
                                # layer following flattening operation
                                elif class_name == 'dense':
                                    if (permute_layer_name is not None) and (tlayer.name == permute_layer_name):
                                        tensor_out = np.zeros(tensor_in.shape)
                                        for jj in range(tensor_out.shape[0]):
                                            tensor_out[jj, :] = tensor_in[perm_index[jj], :]
                                    else:  # not following flattening, just copy
                                        tensor_out = tensor_in.copy()

                                    # mimic Caffe layout
                                    tensor_out = np.transpose(tensor_out, (1, 0))

                        # save weight in format amenable to SAS
                        dset_name = generate_dataset_name(tlayer, wgt_idx)
                        wgt_idx = wgt_idx + 1
                        new_weight_names.append(dset_name)
                        g_out.create_dataset(dset_name, data=tensor_out)

                # update weight names
                g_out.attrs['weight_names'] = new_weight_names

    except ValueError as err_msg:
        print(err_msg)

    finally:
        # close files
        f_out.close()
        f_in.close()
        
    return use_gpu

def write_keras_hdf5(model, rnn_support, hdf5_out):
    '''
    Generate an HDF5 file with trained model parameters given a Keras definition

    Parameters
    ----------
    model : Keras model
       Keras deep learning model
    rnn_support : boolean
       Indicates whether importing RNN models is supported
    hdf5_out : string
       Fully qualified file name of SAS-compatible HDF5 file

    '''
    # open output file
    try:
        f_out = h5py.File(hdf5_out, 'w')
    except IOError:
        raise DLPyError('The specified file cannot be written: ' + hdf5_out)

    model_type = None
    use_gpu = None
    try:
        # determine model type
        # NOTE: must check ALL RNN layers to determine
        #       whether model must run on GPU
        gpu_layers = []
        cpu_layers = []
        for layer in model.layers:
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                # check for RNN layers
                if class_name in rnn_layer_classes:
                    model_type = 'RNN'
                    image_data_format = None
                    if class_name in rnn_gpu_layer_classes:
                        gpu_layers.append(True)
                    elif class_name in rnn_cpu_layer_classes:
                        cpu_layers.append(True)

        # verify that model is supported by SAS Deep Learning
        if model_type == 'RNN':
            if rnn_support:
                if (len(gpu_layers) > 0) and (len(cpu_layers) == 0):
                    use_gpu = True
                elif (len(gpu_layers) == 0) and (len(cpu_layers) > 0):
                    use_gpu = False
                elif (len(gpu_layers) > 0) and (len(cpu_layers) > 0):
                    raise DLPyError('A mixture of CPU and GPU layers was detected. '
                                    'This is not supported by SAS Deep Learning.')
            else:
                raise DLPyError('RNN model detected: your Viya deployment does not support '
                                'importing an RNN model.')
                
        if model_type is None:
            found_cnn_layer = False
            for layer in model.layers:
                class_name, sublayers = remove_layer_wrapper(layer)
                for tlayer in sublayers:
                    # check for CNN layers
                    if class_name in conv_layer_classes:
                        model_type = 'CNN'
                        image_data_format = K.image_data_format()
                        found_cnn_layer = True
                
                if found_cnn_layer:
                    break

        if model_type is None:
            raise DLPyError('Only RNN and CNN models are currently supported.')

        # determine layers with weights
        filtered_layers = []
        filtered_layer_names = []
        for layer in model.layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)
                filtered_layer_names.append(layer.name)
                
        # determine permutation vector associated with flattening layer (if it exists)
        if model_type == 'CNN':
            flatten_layer_index = -1
            index = 0
            for layer in model.layers:
                if layer.__class__.__name__.lower() == 'flatten':
                    flatten_layer_index = index
                    break
                index = index + 1

            if flatten_layer_index != -1:
                layer = model.layers[flatten_layer_index]
                permute_layer_name = model.layers[flatten_layer_index + 1].name
                if image_data_format == 'channels_first':
                    C, H, W = (layer.input_shape)[1:]
                else:
                    H, W, C = (layer.input_shape)[1:]
                N = (layer.output_shape)[1]
                perm_index = [0] * N
                if image_data_format == 'channels_last':
                    ii = 0
                    for cc in range(C):
                        for hh in range(H):
                            for ww in range(W):
                                perm_index[ii] = hh * W * C + ww * C + cc
                                ii = ii + 1
                else:
                    for nn in range(N):
                        perm_index[nn] = nn
            else:
                perm_index = []
                permute_layer_name = None
        else:
            perm_index = []
            permute_layer_name = None

        # populate attributes with layer names
        attrib_layer_names = []
        for name in filtered_layer_names:
            layer = model.get_layer(name=name)
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                attrib_layer_names.append(tlayer.name)
                            
        f_out.attrs['layer_names'] = [replace_forward_slash(l).encode('utf8') for l in attrib_layer_names]            
        # let Keras read weights, reformat, and write to SAS-compatible file
        for k, layer in enumerate(filtered_layers):
            symbolic_weights = layer.weights
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                #weight_names.append(name.encode('utf8'))
                weight_names.append(name)

            # layer modification from here:
            new_weight_names = []

            if len(weight_values) != len(symbolic_weights):
                raise ValueError('Layer #' + str(k) +
                                 ' (named "' + layer.name +
                                 '" in the current model) was found to '
                                 'correspond to layer ' + name +
                                 ' in the saved file. '
                                 'However the new layer ' + layer.name +
                                 ' expects ' + str(len(symbolic_weights)) +
                                 ' weights, but the saved weights have ' +
                                 str(len(weight_values)) +
                                 ' elements.')
                                 
            # create CPU-compatible layer
            cpu_layer = create_cpu_compatible_layer(layer, model_type)
                                 
            # read/write weights
            class_name, sublayers = remove_layer_wrapper(layer)
            for tlayer in sublayers:
                g_out = f_out.create_group(replace_forward_slash(tlayer.name))
                new_weight_names = []
                wgt_idx = 0
            
                for ii,wgt_name in enumerate(weight_names):
                    if tlayer.name in wgt_name:
                        if type(weight_values[ii]) == np.ndarray:
                            tensor_in = weight_values[ii]
                        else:
                            tensor_in = np.zeros(weight_values[ii].shape,
                                                dtype=weight_values[ii].dtype)
                            weight_values[ii].read_direct(tensor_in)

                        # permute axes as needed to conform to SAS deep
                        # learning "channels first" format
                        if (image_data_format is not None) and (image_data_format == 'channels_first'):
                            # format: (C,fdim1, fdim2, fdim3) ==> (C,fdim3,fdim1,fdim2)
                            if len(tensor_in.shape) == 4:
                                tensor_out = np.transpose(tensor_in, (0, 3, 1, 2))
                            else:
                                tensor_out = tensor_in.copy()
                        else:
                            # "channels last" format or not image processing problem
                            
                            # process RNN layers first
                            if class_name in rnn_layer_classes:
                                cpu_class_name, cpu_sublayers = remove_layer_wrapper(cpu_layer)
                                if (len(tensor_in.shape) == 1) and (class_name != cpu_class_name):
                                    tensor_out = np.tile(0.5 * tensor_in, 2)
                                else:
                                    tensor_out = tensor_in.copy()
                            # not an RNN layer, but this is a vector - nothing to permute
                            elif len(tensor_in.shape) == 1:
                                tensor_out = tensor_in.copy()
                            else:
                                # permute Conv2D tensor to "channels_first" format
                                if class_name == 'conv2d':
                                    tensor_out = np.transpose(tensor_in, (3, 2, 0, 1))
                                # have to account for neuron ordering in first dense
                                # layer following flattening operation
                                elif class_name == 'dense':
                                    if (permute_layer_name is not None) and (tlayer.name == permute_layer_name):
                                        tensor_out = np.zeros(tensor_in.shape)
                                        for jj in range(tensor_out.shape[0]):
                                            tensor_out[jj, :] = tensor_in[perm_index[jj], :]
                                    else:  # not following flattening, just copy
                                        tensor_out = tensor_in.copy()

                                    # mimic Caffe layout
                                    tensor_out = np.transpose(tensor_out, (1, 0))

                        # save weight in format amenable to SAS
                        dset_name = generate_dataset_name(tlayer, wgt_idx)
                        wgt_idx = wgt_idx + 1
                        new_weight_names.append(dset_name)
                        g_out.create_dataset(dset_name, data=tensor_out)

                # update weight names
                g_out.attrs['weight_names'] = new_weight_names

    except ValueError as err_msg:
        print(err_msg)

    finally:
        # close files
        f_out.close()

    return use_gpu

def generate_dataset_name(layer, index):
    '''
    Generate data set names consistent with names generated by Keras models

    Parameters
    ----------
    layer : Layer
       Current layer definition
    index : int
       Data set index

    Returns
    -------
    UTF-8 encoded data set name

    '''
    layer_class_name = layer.__class__.__name__.lower()
    if layer_class_name in ['conv2d', 'dense']:
        template_names = ['kernel:0', 'bias:0']
    elif layer_class_name == 'batchnormalization':
        template_names = ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0']
    elif layer_class_name in ['simplernn', 'lstm', 'gru', 'cudnnlstm', 'cudnngru']:
        template_names = ['kernel:0', 'recurrent_kernel:0','bias:0']
    else:
        raise ValueError('Unable to translate layer weight name for layer = ' + layer.name)

    dataset_name = replace_forward_slash(layer.name) + '/' + template_names[index]
    return dataset_name.encode('utf8')
