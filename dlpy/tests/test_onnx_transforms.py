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


class TestTransformer(unittest.TestCase):
    def test_transformer1(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        node1 = helper.make_node(
            'Identity',
            inputs=['in'],
            outputs=['out'],
            name='Identity_1'
        )

        graph_ = helper.make_graph(
            nodes=[node1],
            name='',
            inputs=[],
            outputs=[],
            initializer=[]
        )
        graph = OnnxGraph.from_onnx(graph_)

        pattern = OpTypePattern('Identity')
        transform = Transformer(pattern)
        self.assertTrue(transform.match(graph.node[0]))

    def test_transformer2(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        nodes = [
            helper.make_node('Unsqueeze',
                inputs=['unsqueeze_in'],
                outputs=['unsqueeze_out'],
                name='Unsqueeze_1',
                axes=[1, 2]
            ),
            helper.make_node('Mul',
                inputs=['unsqueeze_out', 'mul1'],
                outputs=['mul_out'],
                name='Mul_2'
            ),
            helper.make_node('Add',
                inputs=['mul_out', 'add1'],
                outputs=['add_out'],
                name='Add_3'
            )
        ]

        graph_ = helper.make_graph(
            nodes=nodes,
            name='',
            inputs=[],
            outputs=[],
            initializer=[]
        )
        graph = OnnxGraph.from_onnx(graph_)
        transform = Transformer()
        pattern = OpTypePattern('Unsqueeze',
                                outputs=[OpTypePattern('Mul', outputs=['Add'])])

        self.assertTrue(transform.match(graph.node[0], pattern))
        self.assertFalse(transform.match(graph.node[1], pattern))

    def test_transformer3(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        nodes = [
            helper.make_node('Unsqueeze',
                inputs=['unsqueeze_in'],
                outputs=['unsqueeze_out'],
                name='Unsqueeze_1',
                axes=[1, 2]
            ),
            helper.make_node('Mul',
                inputs=['unsqueeze_out', 'mul1'],
                outputs=['mul_out'],
                name='Mul_2'
            ),
            helper.make_node('Add',
                inputs=['mul_out', 'add1'],
                outputs=['add_out'],
                name='Add_3'
            ),
        ]

        graph_ = helper.make_graph(
            nodes=nodes,
            name='',
            inputs=[],
            outputs=[],
            initializer=[]
        )
        graph = OnnxGraph.from_onnx(graph_)

        pattern = OpTypePattern('Add', name='op3')
        pattern = OpTypePattern('Mul', name='op2', outputs=[pattern])
        pattern = OpTypePattern('Unsqueeze', name='op1', outputs=[pattern])

        transform = Transformer(pattern)
        mapping = transform.get_mapping(graph.node[0])

        self.assertEqual(len(mapping), 3)
        self.assertEqual(mapping['op3'].name, 'Add_3')
        self.assertEqual(mapping['op2'].name, 'Mul_2')
        self.assertEqual(mapping['op1'].name, 'Unsqueeze_1')

    def test_init_unsqueeze(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        unsqueeze_1 = helper.make_node(
            'Unsqueeze',
            inputs=['unsqueeze_in'],
            outputs=['unsqueeze_out'],
            name='Unsqueeze_1',
            axes=[1, 2]
        )

        unsqueeze_in = np.random.rand(64).astype('float32')
        init = numpy_helper.from_array(unsqueeze_in, name='unsqueeze_in')
        graph_ = helper.make_graph(
            nodes=[unsqueeze_1],
            name='',
            inputs=[],
            outputs=[],
            initializer=[init]
        )
        graph = OnnxGraph.from_onnx(graph_)
        graph = InitUnsqueeze()(graph)

        self.assertEqual(len(graph.node), 0)
        self.assertTrue('unsqueeze_out' in graph.tensor_dict)
        t = graph.tensor_dict['unsqueeze_out']
        t = np.squeeze(t)
        self.assertTrue(np.array_equal(unsqueeze_in, t))
    
    def test_init_reshape(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        reshape_1 = helper.make_node(
            'Reshape',
            inputs=['reshape_in', 'shape'],
            outputs=['reshape_out'],
            name='Reshape_1'
        )

        reshape_in = np.random.rand(64).astype('float32')
        shape = np.array([8, 8])
        init = numpy_helper.from_array(reshape_in, name='reshape_in')
        shape_init = numpy_helper.from_array(shape, name='shape')
        graph_ = helper.make_graph(
            nodes=[reshape_1],
            name='',
            inputs=[],
            outputs=[],
            initializer=[init, shape_init]
        )
        graph = OnnxGraph.from_onnx(graph_)
        graph = InitReshape()(graph)

        self.assertEqual(len(graph.node), 0)
        self.assertTrue('reshape_out' in graph.tensor_dict)
        t = graph.tensor_dict['reshape_out']
        self.assertEqual(t.shape, (8, 8))

    def test_const_to_initializer(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        values = np.random.randn(5, 5).astype(np.float32)
        const_1 = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['constant_out'],
            name='Constant_1',
            value=onnx.helper.make_tensor(
                name='const_tensor',
                data_type=onnx.TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            )
        )

        graph_ = helper.make_graph(
            nodes=[const_1],
            name='',
            inputs=[],
            outputs=[],
            initializer=[]
        )
        graph = OnnxGraph.from_onnx(graph_)
        graph = ConstToInitializer()(graph)

        self.assertEqual(len(graph.node), 0)
        self.assertTrue('constant_out' in graph.tensor_dict)
        t = graph.tensor_dict['constant_out']
        self.assertTrue(np.array_equal(values, t))

    def test_fuse_mul_add_bn(self):
        try:
            import onnx
            from dlpy.model_conversion.onnx_transforms import (Transformer, OpTypePattern,
                                                               ConstToInitializer,
                                                               InitReshape, InitUnsqueeze,
                                                               FuseMulAddBN)
            from dlpy.model_conversion.onnx_graph import OnnxGraph
            from onnx import helper, numpy_helper
        except:
            unittest.TestCase.skipTest(self, 'onnx package not found')

        nodes = [
            helper.make_node('BatchNormalization',
                inputs=['data', 'bn_scale', 'bn_bias', 'bn_mean', 'bn_var'],
                outputs=['bn_out'],
                name='BatchNormalization_1'
            ),
            helper.make_node('Mul',
                inputs=['bn_out', 'mul1'],
                outputs=['mul_out'],
                name='Mul_2'
            ),
            helper.make_node('Add',
                inputs=['mul_out', 'add1'],
                outputs=['add_out'],
                name='Add_3'
            )
        ]
        
        bn_scale = np.random.rand(64).astype('float32')
        bn_bias = np.random.rand(64).astype('float32')
        bn_mean = np.random.rand(64).astype('float32')
        bn_var = np.random.rand(64).astype('float32')
        mul1 = np.random.rand(64, 1, 1)
        add1 = np.random.rand(64, 1, 1)

        initializer = [
            numpy_helper.from_array(bn_scale, name='bn_scale'),
            numpy_helper.from_array(bn_bias, name='bn_bias'),
            numpy_helper.from_array(bn_mean, name='bn_mean'),
            numpy_helper.from_array(bn_var, name='bn_var'),
            numpy_helper.from_array(mul1, name='mul1'),
            numpy_helper.from_array(add1, name='add1')
        ]

        graph_ = helper.make_graph(
            nodes=nodes,
            name='',
            inputs=[],
            outputs=[],
            initializer=initializer
        )
        graph = OnnxGraph.from_onnx(graph_)
        graph = FuseMulAddBN()(graph)

        self.assertEqual(len(graph.node), 1)
        self.assertEqual(graph.node[0].name, 'BatchNormalization_1')
        s = graph.tensor_dict['bn_scale']
        b = graph.tensor_dict['bn_bias']
        s_ = np.multiply(bn_scale, np.squeeze(mul1))
        b_ = np.multiply(bn_bias, np.squeeze(mul1))
        b_ = np.add(b_, np.squeeze(add1))
        self.assertTrue(np.array_equal(s, s_))
        self.assertTrue(np.array_equal(b, b_))

