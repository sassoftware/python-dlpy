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

from dlpy.model import Model
from dlpy.layers import Layer
from dlpy.utils import DLPyError
import collections


class Network(Model):
    '''
    Model for sequentially building of deep learning models

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object
    layers : list of Layer objects.
        Specifies the layers of the sequential model.
    model_table : string, dict or CAS table, optional
        Specifies the CAS table to store the deep learning model.
        Default : None

    Returns
    -------
    :class:`Sequential`

    '''

    def __init__(self, conn, inputs, outputs, model_table=None):
        Model.__init__(self, conn, model_table=model_table)
        self._map_graph_network(inputs, outputs)

    def _map_graph_network(self, inputs, outputs):
        """propagate all of layers"""
        def build_map(start):
            self.layers.append(start)
            if start.name is None:
                start.count_instances()
                start.name = str(start.__class__.__name__) + '_' + str(type(start).number_of_instances)
            if start in inputs:
                return
            for layer in start.src_layers:
                """if the node is visited, continue"""
                if layer in self.layers:
                    continue
                build_map(layer)
                # set the layer's depth
                layer.depth = 0 if str(layer.__class__.__name__) == 'InputLayer' \
                    else max([i.depth for i in layer.src_layers]) + 1
            return

        if not isinstance(inputs, collections.Iterable):
            inputs = [inputs]
        if any(x.__class__.__name__ != 'InputLayer' for x in inputs):
            raise DLPyError('Input layers should be input layer type.')
        if not isinstance(outputs, collections.Iterable):
            outputs = [outputs]
        if not all(x.can_be_last_layer for x in outputs):
            raise DLPyError('Output layers can only be {}'\
                            .format([i.__name__ for i in Layer.__subclasses__() if i.can_be_last_layer]))
        for layer in outputs:
            build_map(layer)
            layer.depth = max([i.depth for i in layer.src_layers]) + 1

        return

    '''
    compile the network
    parse the network node and process CAS Action
    '''
    def compile(self):
        rt = self._retrieve_('deeplearn.buildmodel',
                             model=dict(name=self.model_name, replace=True), type=self.model_type)

        if rt.severity > 1:
            raise DLPyError('cannot build model, there seems to be a problem.')
        sorted_layers = sorted(self.layers, key = lambda Layer: Layer.depth)

        for layer in sorted_layers:
            if layer.type == 'block':
                # TODO support block
                DLPyError("Block is not supported in network")
            if layer.type == 'transconvo':
                layer.calculate_output_padding()
                del layer.config['output_size']
            option = layer.to_model_params()
            rt = self._retrieve_('deeplearn.addlayer', model = self.model_name, **option)
            if rt.severity > 1:
                raise DLPyError('there seems to be an error while adding the ' + layer.name + '.')
        print('NOTE: Model compiled successfully.')
