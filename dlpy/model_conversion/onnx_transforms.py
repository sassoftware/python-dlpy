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

''' Transforms for ONNX graph '''

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, mapping
from onnx import AttributeProto, TensorProto, GraphProto

from dlpy.model_conversion.onnx_graph import OnnxNode


class OpTypePattern(object):
    def __init__(self, op_type, name=None, outputs=None):
        self._op_type = op_type
        self._name = name
        if outputs is None:
            outputs = []
        self._outputs = [
            output_pattern if isinstance(output_pattern, OpTypePattern) else
            OpTypePattern(output_pattern) for output_pattern in outputs
        ]
    
    @property
    def op_type(self):
        return self._op_type
    
    @property
    def outputs(self):
        return self._outputs
    
    @property
    def name(self):
        return self._name

    
class Transformer(object):
    def __init__(self, pattern=None):
        self.pattern = pattern

    def match(self, node, pattern=None):
        if pattern is None:
            if self.pattern is None:
                raise ValueError('No pattern to match.')
            pattern = self.pattern

        if node.op_type != pattern.op_type:
            return False
        elif pattern.outputs and len(node.children) != len(pattern.outputs):
            return False
        else:
            ret = []
            for child, child_pattern in zip(node.children, pattern.outputs):
                r = self.match(child, child_pattern)
                ret.append(r)
            return all(ret)

    def get_mapping(self, node, pattern=None):
        if pattern is None:
            if self.pattern is None:
                raise ValueError('No pattern to match.')
            pattern = self.pattern
        
        mapping_dict = {}
        def _mapping(node, pattern, mapping_dict):
            if pattern.name is None:
                raise ValueError('Cannot generate mapping dict,'
                                ' OpTypePattern name is None.')
            mapping_dict[pattern.name] = node
            for child, child_pattern in zip(node.children, pattern.outputs):
                _mapping(child, child_pattern, mapping_dict)
            return mapping_dict
        
        return _mapping(node, pattern, mapping_dict)

    def is_eligible(self, graph, node):
        return True

    def run_transform(self, graph, node):
        return graph
    
    def __call__(self, graph):
        matches = filter(lambda x: self.match(x), graph.node)
        ops = filter(lambda x: self.is_eligible(graph, x), matches)
        for op in ops:
            self.run_transform(graph, op)
            graph.connect_nodes()
        return graph


class ConstToInitializer(Transformer):
    ''' Remove constant ops '''
    def __init__(self):
        pattern = OpTypePattern('Constant')
        super(ConstToInitializer, self).__init__(pattern)
    
    def run_transform(self, graph, node):
        tensor = numpy_helper.to_array(node.attrs['value'])
        graph.tensor_dict[node.output[0]] = tensor

        # remove the constant op
        graph.remove_node(node.name)
        
        return graph


class InitReshape(Transformer):
    ''' Remove reshape ops '''
    def __init__(self):
        pattern = OpTypePattern('Reshape')
        super(InitReshape, self).__init__(pattern)
    
    def is_eligible(self, graph, node):
        if node.input[0] in graph.tensor_dict:
            return True
        return False
    
    def _get_shape(self, graph, node):
        ''' Get reshape op's shape '''
        name = node.input[1]
        shape = [int(x) for x in graph.tensor_dict[name].flatten()]
        return tuple(shape)

    def run_transform(self, graph, node):
        shape = self._get_shape(graph, node)
        tensor = graph.tensor_dict[node.input[0]]
        graph.tensor_dict[node.output[0]] = tensor.reshape(shape)

        # remove reshape op
        graph.remove_node(node.name)

        return graph


class InitUnsqueeze(Transformer):
    ''' Remove unsqueeze ops '''
    def __init__(self):
        pattern = OpTypePattern('Unsqueeze')
        super(InitUnsqueeze, self).__init__(pattern)
    
    def is_eligible(self, graph, node):
        if node.input[0] in graph.tensor_dict:
            return True
        return False
    
    def _unsqueeze(self, tensor, axes):
        _shape = list(tensor.shape)
        new_dim = len(tensor.shape) + len(axes)
        unsqueezed_shape = [1] * new_dim 
        for i in range(new_dim):
            if i not in axes:
                unsqueezed_shape[i] = _shape.pop(0) 

        return tensor.reshape(tuple(unsqueezed_shape))

    def run_transform(self, graph, node):
        axes = node.attrs['axes']
        tensor = graph.tensor_dict[node.input[0]]
        graph.tensor_dict[node.output[0]] = self._unsqueeze(tensor, axes)

        # remove unsqueeze op
        graph.remove_node(node.name)

        return graph


class FuseMulAddBN(Transformer):
    ''' Fuse Mul + Add into BN '''
    def __init__(self):
        add = OpTypePattern('Add', name='add')
        mul = OpTypePattern('Mul', name='mul', outputs=[add])
        bn = OpTypePattern('BatchNormalization', name='bn', outputs=[mul])
        super(FuseMulAddBN, self).__init__(bn)

    def is_eligible(self, graph, node):
        mapping = self.get_mapping(node)
        bn, mul, add = mapping['bn'], mapping['mul'], mapping['add']

        if bn.attrs.get('spatial') is not None and bn.attrs['spatial'] != 1:
            return False
        if (mul.input[0] not in graph.tensor_dict and
            mul.input[1] not in graph.tensor_dict):
            return False
        if (add.input[0] not in graph.tensor_dict and
            add.input[1] not in graph.tensor_dict):
            return False

        t = graph.tensor_dict
        scale = t[bn.input[1]]
        bias = t[bn.input[2]]
        _mul_tensor = t.get(mul.input[0], t[mul.input[1]])
        mul_tensor = np.squeeze(_mul_tensor)
        _add_tensor = t.get(add.input[0], t[add.input[1]])
        add_tensor = np.squeeze(_add_tensor)

        if mul_tensor.shape != scale.shape or mul_tensor.shape != bias.shape:
            if mul_tensor.shape != (1,) and mul_tensor.shape != ():
                return False
        
        if add_tensor.shape != bias.shape:
            if add_tensor.shape != (1,) and add_tensor.shape != ():
                return False
        
        return True
    
    def run_transform(self, graph, node):
        mapping = self.get_mapping(node)
        bn, mul, add = mapping['bn'], mapping['mul'], mapping['add']

        t = graph.tensor_dict
        scale = t[bn.input[1]]
        bias = t[bn.input[2]]
        _mul_tensor = t.get(mul.input[0], t[mul.input[1]])
        mul_tensor = np.squeeze(_mul_tensor)
        _add_tensor = t.get(add.input[0], t[add.input[1]])
        add_tensor = np.squeeze(_add_tensor)

        # multiply scale and bias
        t[bn.input[1]] = np.multiply(scale, mul_tensor)
        _bias = np.multiply(bias, mul_tensor)
        t[bn.input[2]] = np.add(_bias, add_tensor)

        # connect output of bn to output of add
        bn.output[0] = add.output[0]

        # remove mul and add nodes
        graph.remove_node(mul.name)
        graph.remove_node(add.name)

        return graph

