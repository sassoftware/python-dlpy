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

import unittest
import numpy as np


class TestGraph(unittest.TestCase):
    def _generate_graph1(self):
        try:
            from onnx import helper, numpy_helper, TensorProto
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        input0 = helper.make_tensor_value_info('data0',
                                               TensorProto.FLOAT,
                                               [1, 3, 224, 224])

        input1 = helper.make_tensor_value_info('conv0',
                                               TensorProto.FLOAT,
                                               [64, 3, 7, 7])
        
        output0 = helper.make_tensor_value_info('output0',
                                                TensorProto.FLOAT,
                                                [1, 64, 122, 122])

        conv_op = helper.make_node('Conv',
                                   inputs=['data0', 'conv0'],
                                   outputs=['output0'],
                                   kernel_shape=[7, 7],
                                   pads=[3, 3, 3, 3],
                                   strides=[2, 2])


        conv0 = np.random.rand(64, 3, 7, 7).astype('float32')
        init0 = numpy_helper.from_array(conv0,
                                        name='conv0')

        graph = helper.make_graph(
            nodes=[conv_op],
            name='',
            inputs=[input0, input1],
            outputs=[output0],
            initializer=[init0]
        )
        return graph

    def test_graph1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        self.assertEqual(len(graph.node), 1)
        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(len(graph.input), 2)
        self.assertEqual(len(graph.output), 1)
        self.assertEqual(len(graph.uninitialized), 1)

        self.assertEqual(graph.node[0].name, 'Conv_0')
        self.assertTrue(not graph.node[0].parents)
        self.assertTrue(not graph.node[0].children)

        self.assertEqual(graph.initializer[0].name, 'conv0')
        self.assertEqual(graph.input[0].name, 'data0')
        self.assertEqual(graph.input[1].name, 'conv0')
        self.assertEqual(graph.output[0].name, 'output0')

        self.assertTrue('conv0' in graph.tensor_dict)
        self.assertEqual(graph.uninitialized[0].name, 'data0')

    def test_graph_connection(self):
        try:
            import onnx
            from onnx import helper, numpy_helper, TensorProto
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        input0 = helper.make_tensor_value_info('data0',
                                               TensorProto.FLOAT,
                                               [1, 3, 224, 224])

        input1 = helper.make_tensor_value_info('conv0',
                                               TensorProto.FLOAT,
                                               [64, 3, 7, 7])
        
        output0 = helper.make_tensor_value_info('output0',
                                                TensorProto.FLOAT,
                                                [1, 64, 122, 122])

        conv_op = helper.make_node('Conv',
                                   inputs=['data0', 'conv0'],
                                   outputs=['conv_out'],
                                   kernel_shape=[7, 7],
                                   pads=[3, 3, 3, 3],
                                   strides=[2, 2])

        identity_op = helper.make_node('Identity',
                                       inputs=['conv_out'],
                                       outputs=['output0'])

        conv0 = np.random.rand(64, 3, 7, 7).astype('float32')
        init0 = numpy_helper.from_array(conv0,
                                        name='conv0')

        graph_ = helper.make_graph(
            nodes=[conv_op, identity_op],
            name='',
            inputs=[input0, input1],
            outputs=[output0],
            initializer=[init0]
        )

        graph = OnnxGraph.from_onnx(graph_)

        self.assertEqual(len(graph.node), 2)

        self.assertEqual(graph.node[0].name, 'Conv_0')
        self.assertTrue(not graph.node[0].parents)
        self.assertEqual(len(graph.node[0].children), 1)
        self.assertEqual(graph.node[0].children[0].name, 'Identity_1')
        self.assertTrue('conv0' in graph.node[0].tensors)

        self.assertEqual(graph.node[1].name, 'Identity_1')
        self.assertEqual(len(graph.node[1].parents), 1)
        self.assertEqual(graph.node[1].parents[0].name, 'Conv_0')
        self.assertTrue(not graph.node[1].children)
        self.assertTrue(not graph.node[1].tensors)

    def test_get_node(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        self.assertEqual(graph.get_node('Conv_0').name, 'Conv_0')
        self.assertEqual(graph.get_node_index('Conv_0'), 0)
        self.assertEqual(graph.get_node('abcdef'), None)
        self.assertEqual(graph.get_node_index('abcdef'), None)

    def test_remove_node(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        graph.remove_node('Conv_0')
        self.assertTrue(not graph.node)
    
    def test_remove_node1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        graph.remove_node('abcdef')
        self.assertEqual(len(graph.node), 1)
    
    def test_replace_node(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')


        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        node_ = graph.node[0]
        new_node = helper.make_node('Identity',
                                    inputs=node_.input,
                                    outputs=node_.output,
                                    name='test_node')
        new_node = OnnxNode(new_node)
        graph.replace_node('Conv_0', new_node)

        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].name, 'test_node')
    
    def test_replace_node1(self):
        try:
            import onnx
            from onnx import helper
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')


        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        node_ = graph.node[0]
        new_node = helper.make_node('Identity',
                                    inputs=node_.input,
                                    outputs=node_.output,
                                    name='test_node')
        new_node = OnnxNode(new_node)
        graph.replace_node('abcdef', new_node)

        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].name, 'Conv_0')

    def test_insert_node(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')


        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        new_node = helper.make_node('Identity',
                                    inputs=[],
                                    outputs=[],
                                    name='test_node')
        new_node = OnnxNode(new_node)
        graph.insert_node('Conv_0', new_node)

        self.assertEqual(len(graph.node), 2)
        self.assertEqual(graph.node[1].name, 'test_node')
    
    def test_insert_node1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')


        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        new_node = helper.make_node('Identity',
                                    inputs=[],
                                    outputs=[],
                                    name='test_node')
        new_node = OnnxNode(new_node)
        graph.insert_node('abcdef', new_node)

        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].name, 'Conv_0')

    def test_get_input(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        i = graph.get_input('data0')
        self.assertEqual(i.name, 'data0')
        self.assertEqual([d.dim_value for d in i.type.tensor_type.shape.dim],
                          [1, 3, 224, 224])

    def test_get_input1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        i = graph.get_input('abcdef')
        self.assertEqual(i, None)

    def test_add_input(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import helper, TensorProto
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')


        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        value_info = helper.make_tensor_value_info('data1',
                                                   TensorProto.FLOAT,
                                                   [1, 3, 299, 299])
        graph.add_input(value_info)
        self.assertEqual(len(graph.input), 3)
        self.assertEqual(graph.input[-1], value_info)
    
    def test_replace_input(self):
        try:
            import onnx
            from onnx import helper, TensorProto
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        value_info = helper.make_tensor_value_info('data1',
                                                   TensorProto.FLOAT,
                                                   [1, 3, 299, 299])
        graph.replace_input('data0', value_info)
        self.assertEqual(len(graph.input), 2)
        self.assertEqual(graph.input[0], value_info)

    def test_get_initializer(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        init = graph.get_initializer('conv0')
        conv0 = numpy_helper.to_array(init)

        self.assertEqual(init.name, 'conv0')
        self.assertTrue(np.array_equal(conv0, graph.tensor_dict['conv0']))
    
    def test_get_initializer1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        init = graph.get_initializer('abcdef')
        self.assertEqual(init, None)

    def test_add_initializer(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        conv1 = np.random.rand(64, 3, 7, 7).astype('float32')
        init1 = numpy_helper.from_array(conv1,
                                        name='conv1')
        graph.add_initializer(init1)

        self.assertEqual(len(graph.initializer), 2)
        self.assertEqual(graph.initializer[1], init1)
    
    def test_replace_initializer(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
            from onnx import numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        conv1 = np.random.rand(64, 3, 7, 7).astype('float32')
        init1 = numpy_helper.from_array(conv1,
                                        name='conv1')
        graph.replace_initializer('conv0', init1)

        self.assertEqual(len(graph.initializer), 1)
        self.assertEqual(graph.initializer[0], init1)
    
    def test_clean_init(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        graph.remove_node('Conv_0')
        graph.clean_init()

        self.assertTrue(not graph.input)
        self.assertTrue(not graph.initializer)
        self.assertTrue(not graph.tensor_dict)

    def test_make_model(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_graph import OnnxGraph, OnnxNode
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        graph_ = self._generate_graph1()
        graph = OnnxGraph.from_onnx(graph_)

        model = graph.make_onnx()
        g = model.graph
        self.assertEqual(len(g.node), 1)
        self.assertEqual(g.node[0].name, 'Conv_0')
        self.assertEqual(len(g.input), 2)

