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

''' Classes to hold ONNX graph and node protobufs '''

import onnx
from onnx import helper, numpy_helper, mapping
from onnx import NodeProto


def _convert_onnx_attribute_proto(attr_proto):
    '''
    Convert ONNX AttributeProto into Python object
    '''
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        str_list = list(map(lambda x: str(x, 'utf-8'), str_list))
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))    


class OnnxNode(object):
    ''' 
    Reimplementation of NodeProto from ONNX, but in a form
    more convenient to work with from Python.
    ''' 
    def __init__(self, node):
        '''
        Create OnnxNode from NodeProto

        Parameters
        ----------
        node : NodeProto

        Returns
        -------
        :class:`OnnxNode` object

        '''
        self.name = str(node.name)
        self.op_type = str(node.op_type)
        self.domain = str(node.domain)
        self.attrs = dict([(attr.name,
                            _convert_onnx_attribute_proto(attr))
                           for attr in node.attribute])
        self.input = list(node.input)
        self.output = list(node.output)
        self.node_proto = node
        self.parents = []
        self.children = []
        self.tensors = {}

    def add_child(self, child):
        '''
        Add child node

        Parameters
        ----------
        child : :class:`OnnxNode` object

        '''
        if not isinstance(child, (tuple, list)):
            child = [child]
        child = list(filter(lambda x: x not in self.children, child))
        self.children.extend(child)
        
        for c in child:
            if self not in c.parents:
                c.add_parent(self)

    def add_parent(self, parent):
        '''
        Add OnnxNode parent

        Parameters
        ----------
        parent : :class:`OnnxNode` object

        '''
        if not isinstance(parent, (tuple, list)):
            parent = [parent]
        parent = list(filter(lambda x: x not in self.parents, parent))
        self.parents.extend(parent)

        for p in parent:
            if self not in p.children:
                p.add_child(self)


class OnnxGraph(object):
    '''
    Helper class for holding ONNX graph

    Parameters
    ----------
    graph_def : GraphProto

    Returns
    -------
    :class:`OnnxGraph` object

    '''
    def __init__(self, graph_def):
        self.name = graph_def.name
        self.node = [OnnxNode(n) for n in graph_def.node]
        self.value_info = list(graph_def.value_info)
        self.input = list(graph_def.input)
        self.output = list(graph_def.output)
        self.initializer = list(graph_def.initializer)
        self.tensor_dict = dict([(init.name, numpy_helper.to_array(init))
                                  for init in graph_def.initializer])
        self.uninitialized = [i for i in graph_def.input
                              if i.name not in self.tensor_dict]
    
    def get_node(self, name):
        '''
        Get node by name

        Parameters
        ----------
        name : str
            Name of the node.

        Returns
        -------
        :class:`OnnxNode` object if node is in graph, otherwise None

        '''
        for n in self.node:
            if n.name == name:
                return n
        return None

    def get_node_index(self, name):
        '''
        Get index of node

        Parameters
        ----------
        name : str
            Name of the node.

        Returns
        -------
        int if node is in graph, otherwise None

        '''
        for idx, n in enumerate(self.node):
            if n.name == name:
                return idx
        return None
    
    def remove_node(self, name):
        '''
        Remove node from graph

        Parameters
        ----------
        name : str
            Name of node to be removed.

        '''
        self.node = list(filter(lambda x: x.name != name, self.node))
        self.connect_nodes()

    def replace_node(self, name, node):
        '''
        Replace node in graph

        Parameters
        ----------
        name : str
            Name of node to be replaced.
        node : :class:`OnnxNode` object
            The replacement node.

        '''
        idx = self.get_node_index(name)
        if idx is not None:
            self.node[idx] = node
        self.connect_nodes()
    
    def insert_node(self, name, node):
        '''
        Insert node in graph after named node

        Parameters
        ----------
        name : str
            Name of the node to insert `node` after.
        node : :class:`OnnxNode` object
            The node to insert.

        '''
        idx = self.get_node_index(name)
        if idx is not None:
            self.node.insert(idx+1, node)
        self.connect_nodes()
    
    def get_input(self, name):
        '''
        Get graph input ValueInfoProto

        Parameters
        ----------
        name : str
            Name of the ValueInfoProto.

        Returns
        -------
        :class:`ValueInfoProto` object, or None if not present.

        '''
        for i in self.input:
            if i.name == name:
                return i
        return None

    def add_input(self, value_info):
        '''
        Add new graph input ValueInfoProto

        Parameters
        ----------
        value_info : :class:`ValueInfoProto` object
            ValueInfoProto to add to graph input.

        '''
        if not isinstance(value_info, (list, tuple)):
            value_info = [value_info] 
        self.input.extend(value_info)

    def replace_input(self, name, value_info):
        '''
        Replace a graph input ValueInfoProto

        Parameters
        ----------
        name : str
            Name of ValueInfoProto to be replaced.
        value_info : :class:`ValueInfoProto` object
            The replacement ValueInfoProto.

        '''
        for idx, proto in enumerate(self.input):
            if proto.name == name:
                self.input[idx] = value_info

    def get_initializer(self, name):
        '''
        Get TensorProto from initializer

        Parameters
        ----------
        name : str
            Name of the TensorProto.

        Returns
        -------
        :class:`TensorProto` object, or None if not present.

        '''
        for i in self.initializer:
            if i.name == name:
                return i
        return None

    def add_initializer(self, init):
        '''
        Add TensorProto to initializer

        Parameters
        ----------
        init : :class:`TensorProto` object
            TensorProto to add to initializer.

        '''
        if not isinstance(init, (list, tuple)):
            init = [init]
        self.initializer.extend(init)

    def replace_initializer(self, name, init):
        '''
        Replace TensorProto in initializer

        Parameters
        ----------
        name : str
            Name of TensorProto to be replaced.
        init : :class:`TensorProto` object
            The replacement TensorProto.

        '''
        for idx, proto in enumerate(self.initializer):
            if proto.name == name:
                self.initializer[idx] = init

    def clean_init(self):
        ''' Remove inputs, initializers which are not part of graph '''
        all_inputs = [i for n in self.node for i in n.input]
        self.input = list(filter(lambda x: x.name in all_inputs,
                                 self.input))
        self.initializer = list(filter(lambda x: x.name in all_inputs,
                                       self.initializer))
        self.tensor_dict = {k:v for k,v in self.tensor_dict.items()
                                if k in all_inputs}

    def connect_nodes(self):
        ''' Add parents and children for each node '''
        # mapping from input to nodes
        input_to_node = {}
        for node in self.node:
            # reset any existing links
            node.parents = []
            node.children = []
            for input_ in node.input:
                if input_to_node.get(input_) is None:
                    input_to_node[input_] = []
                if node not in input_to_node[input_]:
                    input_to_node[input_].append(node)

        for node in self.node:
            for output_ in node.output:
                if not input_to_node.get(output_):
                    continue
                node.add_child(input_to_node[output_])
    
    def make_onnx(self):
        ''' Generate ONNX model from current graph '''
        self.clean_init()
        nodes = []
        for node in self.node:
            n = NodeProto()
            n.input.extend(node.input)
            n.output.extend(node.output)
            n.name = node.name
            n.op_type = node.op_type
            n.attribute.extend(
                helper.make_attribute(key, value)
                for key, value in sorted(node.attrs.items())
                )
            nodes.append(n)
        
        inputs = []
        initializer = []
        for k,v in self.tensor_dict.items():
            init = numpy_helper.from_array(v, name=k)
            initializer.append(init)
            value_info = helper.make_tensor_value_info(
                name=k,
                elem_type=mapping.NP_TYPE_TO_TENSOR_TYPE[v.dtype],
                shape=list(v.shape)
            )
            inputs.append(value_info)
        
        graph_ = helper.make_graph(
            nodes=nodes,
            name='dlpy_graph',
            inputs=inputs+self.uninitialized,
            outputs=self.output,
            initializer=initializer
        )

        model = helper.make_model(graph_)
        return model

    @classmethod
    def from_onnx(cls, graph):
        ''' Create a OnnxGraph object from ONNX GraphProto '''
        graph_ = cls(graph)
        # generate names for nodes
        for idx, node in enumerate(graph_.node):
            if not node.name:
                node.name = '{}_{}'.format(node.op_type, idx)
            elif '/' in node.name:
                node.name.replace('/', '_')
        graph_.connect_nodes()
        # add initialized tensors to nodes
        for node in graph_.node:
            for input_ in node.input:
                if input_ in graph_.tensor_dict:
                    node.tensors[input_] = graph_.tensor_dict[input_]
        return graph_

