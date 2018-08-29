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

import sys
import h5py
import numpy as np
from keras import backend as K
try:
    from keras.engine.topology import preprocess_weights_for_loading
except ImportError:
    from keras.engine.saving import preprocess_weights_for_loading


# let Keras read parameters and then transform to format needed for SAS deep learning
# NOTE: modified version of Keras function load_weights_from_hdf5_group()
def write_keras_hdf5_from_file(model, hdf5_in, hdf5_out):
    '''
    Generate an HDF5 file with trained model parameters given a Keras definition

    Parameters
    ----------
    model : Keras model
       Keras deep learning model
    hdf5_in : string
       Fully qualified file name of Keras HDF5 file
    hdf5_out : string
       Fully qualified file name of SAS-compatible HDF5 file

    '''
    # open input/output files
    try:
        f_in = h5py.File(hdf5_in, 'r')
    except IOError:
        sys.exit('File ' + hdf5_in + ' does not exist')

    try:
        f_out = h5py.File(hdf5_out, 'w')
    except IOError:
        sys.exit('File ' + hdf5_out + ' could not be created')

    if 'keras_version' in f_in.attrs:
        original_keras_version = f_in.attrs['keras_version'].decode('utf8')
    else:
        original_keras_version = '1'
    if 'backend' in f_in.attrs:
        original_backend = f_in.attrs['backend'].decode('utf8')
    else:
        original_backend = None

    try:

        image_data_format = K.image_data_format()

        # determine layers with weights
        filtered_layers = []
        for layer in model.layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)

        layer_names = [n.decode('utf8') for n in f_in.attrs['layer_names']]
        filtered_layer_names = []
        for name in layer_names:
            g = f_in[name]
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
        flatten_layer_index = -1
        index = 0
        for layer in model.layers:
            if (layer.__class__.__name__.lower() == 'flatten'):
                flatten_layer_index = index
                break
            index = index + 1

        if (flatten_layer_index != -1):
            layer = model.layers[flatten_layer_index]
            permute_layer_name = model.layers[flatten_layer_index + 1].name
            if (image_data_format == 'channels_first'):
                C, H, W = (layer.input_shape)[1:]
            else:
                H, W, C = (layer.input_shape)[1:]
            N = (layer.output_shape)[1]
            perm_index = [0] * N
            if (image_data_format == 'channels_last'):
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

            # let Keras read weights, reformat, and write to SAS-compatible file
        for k, name in enumerate(layer_names):
            g_in = f_in[name]
            g_out = f_out.create_group(name)
            new_weight_names = []

            weight_names = [n.decode('utf8') for n in g_in.attrs['weight_names']]
            weight_values = [g_in[weight_name] for weight_name in weight_names]
            layer = filtered_layers[k]
            symbolic_weights = layer.weights
            weight_values = preprocess_weights_for_loading(layer,
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

            # read/write weights
            for ii in range(len(weight_names)):
                tensor_in = np.zeros(weight_values[ii].shape,
                                     dtype=weight_values[ii].dtype)
                weight_values[ii].read_direct(tensor_in)

                # permute axes as needed to conform to SAS deep
                # learning "channels first" format
                if ((image_data_format == 'channels_first') or (not perm_index)):
                    # format: (C,fdim1, fdim2, fdim3) ==> (C,fdim3,fdim1,fdim2)
                    if (len(tensor_in.shape) == 4):
                        tensor_out = np.transpose(tensor_in, (0, 3, 1, 2))
                    else:
                        tensor_out = tensor_in.copy()
                else:
                    # "channels last" format
                    # this is a vector - nothing to permute
                    if (len(tensor_in.shape) == 1):
                        tensor_out = tensor_in.copy()
                    else:
                        # permute Conv2D tensor to "channels_first" format
                        if (layer.__class__.__name__ == 'Conv2D'):
                            tensor_out = np.transpose(tensor_in, (3, 2, 0, 1))
                        # have to account for neuron ordering in first dense
                        # layer following flattening operation
                        elif (layer.__class__.__name__ == 'Dense'):
                            if (layer.name == permute_layer_name):
                                tensor_out = np.zeros(tensor_in.shape)
                                for jj in range(tensor_out.shape[0]):
                                    tensor_out[jj, :] = tensor_in[perm_index[jj], :]
                            else:  # not following flattening, just copy
                                tensor_out = tensor_in.copy()

                            # mimic Caffe layout
                            tensor_out = np.transpose(tensor_out, (1, 0))

                            # save weight in format amenable to SAS
                dset_name = generate_dataset_name(layer, ii)
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

    # generate dataset name with template layerName/layerName/weightName:0


def write_keras_hdf5(model, hdf5_out):
    '''
    Generate an HDF5 file with trained model parameters given a Keras definition

    Parameters
    ----------
    model : Keras model
       Keras deep learning model
    hdf5_out : string
       Fully qualified file name of SAS-compatible HDF5 file

    '''
    # open output file
    try:
        f_out = h5py.File(hdf5_out, 'w')
    except IOError:
        sys.exit('File ' + hdf5_out + ' could not be created')

    try:
        image_data_format = K.image_data_format()

        # determine layers with weights
        filtered_layers = []
        for layer in model.layers:
            weights = layer.weights
            if weights:
                filtered_layers.append(layer)

        # determine permutation vector associated with flattening layer (if it exists)
        flatten_layer_index = -1
        index = 0
        for layer in model.layers:
            if (layer.__class__.__name__.lower() == 'flatten'):
                flatten_layer_index = index
                break
            index = index + 1

        if (flatten_layer_index != -1):
            layer = model.layers[flatten_layer_index]
            permute_layer_name = model.layers[flatten_layer_index + 1].name
            if (image_data_format == 'channels_first'):
                C, H, W = (layer.input_shape)[1:]
            else:
                H, W, C = (layer.input_shape)[1:]
            N = (layer.output_shape)[1]
            perm_index = [0] * N
            if (image_data_format == 'channels_last'):
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

            # let Keras read weights, reformat, and write to SAS-compatible file
        for k, layer in enumerate(filtered_layers):
            # g_in = f_in[name]
            g_out = f_out.create_group(layer.name)
            symbolic_weights = layer.weights
            weight_values = K.batch_get_value(symbolic_weights)
            weight_names = []
            for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                if hasattr(w, 'name') and w.name:
                    name = str(w.name)
                else:
                    name = 'param_' + str(i)
                weight_names.append(name.encode('utf8'))

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
            # read/write weights
            for ii in range(len(weight_names)):
                tensor_in = weight_values[ii]

                # permute axes as needed to conform to SAS deep
                # learning "channels first" format
                if ((image_data_format == 'channels_first') or (not perm_index)):
                    # format: (C,fdim1, fdim2, fdim3) ==> (C,fdim3,fdim1,fdim2)
                    if (len(tensor_in.shape) == 4):
                        tensor_out = np.transpose(tensor_in, (0, 3, 1, 2))
                    else:
                        tensor_out = tensor_in.copy()
                else:
                    # "channels last" format
                    # this is a vector - nothing to permute
                    if (len(tensor_in.shape) == 1):
                        tensor_out = tensor_in.copy()
                    else:
                        # permute Conv2D tensor to "channels_first" format
                        if (layer.__class__.__name__ == 'Conv2D'):
                            tensor_out = np.transpose(tensor_in, (3, 2, 0, 1))
                        # have to account for neuron ordering in first dense
                        # layer following flattening operation
                        elif (layer.__class__.__name__ == 'Dense'):
                            if (layer.name == permute_layer_name):
                                tensor_out = np.zeros(tensor_in.shape)
                                for jj in range(tensor_out.shape[0]):
                                    tensor_out[jj, :] = tensor_in[perm_index[jj], :]
                            else:  # not following flattening, just copy
                                tensor_out = tensor_in.copy()

                            # mimic Caffe layout
                            tensor_out = np.transpose(tensor_out, (1, 0))

                            # save weight in format amenable to SAS
                dset_name = generate_dataset_name(layer, ii)
                new_weight_names.append(dset_name)

                g_out.create_dataset(dset_name, data=tensor_out)

            # update weight names
            g_out.attrs['weight_names'] = new_weight_names
    except ValueError as err_msg:
        print(err_msg)

    finally:
        # close files
        f_out.close()
    # generate dataset name with template layerName/layerName/weightName:0


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
    if (layer_class_name in ['conv2d', 'dense']):
        template_names = ['kernel:0', 'bias:0']
    elif (layer_class_name == 'batchnormalization'):
        template_names = ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0']
    else:
        sys.exit('Unable to translate layer weight name for layer = ' + layer.name)

    dataset_name = layer.name + '/' + template_names[index]
    return dataset_name.encode('utf8')


#########################################################################################
if __name__ == '__main__':
    sys.exit('ERROR: this module cannot be invoked from the command line')
