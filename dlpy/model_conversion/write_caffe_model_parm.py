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

''' Supporting functions for caffe model conversion '''

import h5py


class HDF5WriteError(IOError):
    '''
    Used to indicate an error in writing HDF5 file

    '''


# write Caffe model parameters in HDF5 format
def write_caffe_hdf5(net, layer_list, file_name):
    '''
    Generate a SAS deep learning model from Caffe definition

    Parameters
    ----------
    net : Net
       Caffe network object - used for obtaining parameters (weights/biases/etc.)
    layer_list : list-of-CompositeLayer
       List of layers.  Parameter for these layers must be written in HDF5 format
    file_name : string
       Fully qualified file name of SAS-compatible HDF5 file (*.caffemodel.h5)

    '''

    # open output file
    fout = h5py.File(file_name, 'w')

    # create base group
    g = fout.create_group('data')

    try:

        # write output file
        params = net.params
        for name in params.keys():

            # associate scale layers with batchnorm layers
            match = False
            prop_name = name
            for clayer in layer_list:
                # search for scale layer
                for ii in range(len(clayer.related_layers)):
                    if ((clayer.related_layers[ii].type.lower() == 'scale') and
                            (name == clayer.related_layers[ii].name) and
                            (not match)):
                        prop_name = clayer.layer_parm.name + '_scale'
                        match = True

                if match:
                    break

            if not match:
                prop_name = name

            # open/create group
            cur_group = g.create_group(prop_name)

            # data set indexed by blob number
            for ii in range(len(params[name])):
                blob = params[name][ii].data

                # save parameters in HDF5 format
                dset = cur_group.create_dataset(str(ii), data=blob)

        # every layer in layer_list must have a corresponding group in the parameter
        # file.  Add dummy group(s) for layers that don't have parameters
        for layer in layer_list:
            if (layer.layer_parm.name not in params.keys()):
                cur_group = g.create_group(layer.layer_parm.name)

    except HDF5WriteError as err_str:
        print(err_str)

    finally:
        # close file
        fout.close()
