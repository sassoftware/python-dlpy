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

''' Sequential object for deep learning '''

from __future__ import print_function
from dlpy.model import Model
from dlpy.layers import Layer
from dlpy.utils import DLPyError
from dlpy.blocks import Bidirectional


class Sequential(Model):
    '''
    Model for sequentially building of deep learning models

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    layers : list-of-Layers, optional
        Specifies the layers of the sequential model.
    model_table : string or dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default: None

    Returns
    -------
    :class:`Sequential`

    '''

    def __init__(self, conn, layers=None, model_table=None):
        super(Sequential, self).__init__(conn=conn, model_table=model_table)

        self.layers_dict = {}
        if layers is None:
            self.layers = []
        elif type(layers) is list or type(layers) is set or type(layers) is tuple:
            self.layers = layers
            for layer in self.layers:
                if layer.name is not None:
                    self.layers_dict[layer.name] = layer
            if len(layers) > 0 and isinstance(layers[-1], Layer) and layers[-1].can_be_last_layer:
                self.compile()
            else:
                raise DLPyError('layers has to be a list of layer(s).')
        else:
            raise DLPyError('layers has to be a list of layer(s).')

    def add(self, layer):
        '''
        Add layer(s) to model

        Parameters
        ----------
        layer : Layer or list-of-Layers
            Specifies the layer to be added

        '''
        self.layers.append(layer)
        if isinstance(layer, Layer):
            if layer.name is not None:
                self.layers_dict[layer.name] = layer

            if layer.type == 'recurrent':
                self.model_type = 'RNN'

            print('NOTE: '+layer.type_desc+' added.')

            if layer.can_be_last_layer:
                self.compile()

        if isinstance(layer, Bidirectional):

            self.model_type = 'RNN'

            if layer.src_layers is not None:
                new_src_layers = []
                for l in layer.src_layers:
                    if not isinstance(l, Layer):
                        if l in self.layers_dict:
                            new_src_layers.append(self.layers_dict[l])
                        else:
                            raise DLPyError('cannot find the layer named: '+l)
                    else:
                        new_src_layers.append(l)
                layer.src_layers = new_src_layers

    def pop(self, loc=-1):
        '''
        Delete layer(s) from model and return it

        Parameters
        ----------
        loc : int
            Specifies the index of the layer in the model

        Returns
        -------
        :class:`Layer`

        '''
        if len(self.layers) > 0:
            return self.layers.pop(loc)

    def switch(self, loc1, loc2):
        '''
        Switch the order of two layers in the model.

        Parameters
        ----------
        loc1 : int
            Specifies the index of the first layer
        loc2 : int
            Specifies the index of the second layer

        '''
        self.layers[loc1], self.layers[loc2] = self.layers[loc2], self.layers[loc1]

    def compile(self):
        ''' Convert the layer objects into CAS action parameters '''

        if len(self.layers) == 0:
            raise DLPyError('There is no layers in the model yet.')

        if len(self.layers) > 0 and self.layers[0].type != 'block' and self.layers[0].type != 'input':
            raise DLPyError('The first layer of the model must be an input layer.')

        rt = self._retrieve_('deeplearn.buildmodel',
                             model=dict( name=self.model_name, replace=True), type=self.model_type)

        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('cannot build model, there seems to be a problem.')

        input_num = 1
        conv_num = 1
        fc_num = 1
        bn_num = 1
        concat_num = 1
        scale_num = 1
        reshape_num = 1

        detect_num = 1
        output_num = 1
        keypoints_num = 1

        block_num = 1
        compiled_layers = []
        layer_count = 0

        layer_counts = {}

        for layer in self.layers:
            if layer.type == 'block':

                if isinstance(layer, Bidirectional):
                    output_layer = layer.layers[-2:]
                    options = layer.compile(block_num = block_num)
                else:
                    options = layer.compile(src_layer = output_layer, block_num = block_num)
                    output_layer = layer.layers[-1]

                if isinstance(layer, Bidirectional):
                    block_num += layer.n_blocks
                else:
                    block_num += 1

                for item in layer.layers:
                    compiled_layers.append(item)
                    layer_count += 1

                for option in options:
                    rt = self._retrieve_('deeplearn.addlayer', model=self.model_name, **option)
                    if rt.severity > 1:
                        if layer.name is not None:
                            raise DLPyError('there seems to be an error while adding the '+layer.name+'.')
                        else:
                            raise DLPyError('there seems to be an error while adding a layer.')
            else:
                if isinstance(layer, Layer):

                    layer_counts[layer.type] = layer_counts.get(layer.type, 0) + 1

                    # Name each layer of the model.
                    if layer.type == 'input':
                        if layer.name is None:
                            layer.format_name(local_count=layer_counts[layer.type])

                            #layer.format_name(local_count=input_num)
                            #input_num += 1
                    else:
                        if layer.src_layers is None:
                            if type(output_layer) is list:
                                layer.src_layers = output_layer
                            else:
                                layer.src_layers = [output_layer]

                        '''if layer.type == 'convo':
                            if layer.name is None:
                                layer.format_name(block_num, conv_num)
                                conv_num += 1
                        elif layer.type == 'pool':
                            if layer.name is None:
                                layer.format_name()
                                block_num += 1
                                conv_num = 1
                                bn_num = 1
                                concat_num = 1
                                scale_num = 1
                                reshape_num = 1
                        elif layer.type == 'fc':
                            if layer.name is None:
                                layer.format_name()
                                fc_num += 1
                        elif layer.type == 'batchnorm':
                            if layer.name is None:
                                layer.format_name(block_num, bn_num)
                                bn_num += 1
                        elif layer.type == 'concat':
                            if layer.name is None:
                                layer.format_name(block_num, concat_num)
                                concat_num += 1
                        elif layer.type == 'scale':
                            if layer.name is None:
                                layer.format_name(block_num, scale_num)
                                scale_num += 1
                        elif layer.type == 'reshape':
                            if layer.name is None:
                                layer.format_name(block_num, reshape_num)
                                reshape_num += 1
                        elif layer.type == 'output':
                            if layer.name is None:
                                layer.format_name(local_count=output_num)
                        elif layer.type == 'keypoints':
                            if layer.name is None:
                                layer.format_name(local_count=keypoints_num)
                                keypoints_num += 1
                        elif layer.type == 'detection':
                            if layer.name is None:
                                layer.format_name(local_count=detect_num)
                                detect_num += 1
                        else:
                            if layer.name is None:
                                layer.format_name()
                        '''
                        if layer.name is None:
                            layer.format_name(local_count=layer_counts[layer.type])
                else:
                    raise DLPyError(layer+' is not a type of layer.')

                option = layer.to_model_params()
                compiled_layers.append(layer)
                layer_count += 1
                output_layer = layer

                rt = self._retrieve_('deeplearn.addlayer', model=self.model_name, **option)
                if rt.severity > 1:
                    for m in rt.messages:
                        print(m)
                    raise DLPyError('there seems to be an error while adding the '+layer.name+'.')

        print('NOTE: Model compiled successfully.')
        self.layers = compiled_layers
        self.num_params = self.count_params()
