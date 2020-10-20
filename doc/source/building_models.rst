.. Copyright SAS Institute

.. currentmodule:: dlpy
.. _build_models:

***************
Building Models
***************

DLPy offers you several options to build neural networks, including building them from
scratch or using architectures already defined in the DLPy API. Similarly, you can train
your models from scratch or you can load pre-trained weights. Regardless of your approach,
it helps to know the difference between the Sequential Model and the Functional API. This
document serves as an introduction to these topics, but for more detail see the examples
on `GitHub <https://github.com/sassoftware/python-dlpy/tree/master/examples/functional_model>`_.

**Note:** All examples on this page assume that you have a CAS server started and have
created a CAS session in swat named sess.

Sequential Model
----------------
.. currentmodule:: dlpy.sequential

A :class:`Sequential` model is a modular way to build a custom neural network. A
Sequential model acts like a stack of layers or blocks (e.g. a ResNet block).

.. ipython:: python
   :suppress:

   import os
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = os.environ.get('CASUSER', None)
   password = os.environ.get('CASPASSWORD', None)
   import swat
   sess = swat.CAS(host, port, userid, password)

The following example illustrates how you can use the Sequential Model in order to
create a LeNet architecture.

.. ipython:: python

   from dlpy import Sequential, InputLayer, Conv2d, Pooling, Dense, OutputLayer
    sequential_model = Sequential(conn=sess, model_table="Simple_CNN")
    sequential_model.add(InputLayer(n_channels=1, width=28, height=28))
    sequential_model.add(Conv2d(n_filters=6, width=5, height=5, stride=1))
    sequential_model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    sequential_model.add(Conv2d(n_filters=16, width=5, height=5, stride=1))
    sequential_model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    sequential_model.add(Dense(n=120))
    sequential_model.add(Dense(n=84))
    sequential_model.add(OutputLayer(n=10))

The Sequential model is best used when your network
architecture can be easily specified as a stack of modular elements. Although
the Sequential model can be used to specify architectures with multi-input and multi-output layers,
(see the src_layers parameter of the :class:`dlpy.layers.Layer` class for details),
architectures with complex connections are often better suited for the Functional API.

An additional option is to generate a Functional
model from an existing Sequential model. This can be useful to quickly create
variations of the same basic architecture. To generate a Functional model from a
Sequential model, use :func:`Sequential.to_functional_model`. Then customize
the Functional model with the Functional API as detailed in `Functional API`_.

.. ipython:: python

   functional_model = sequential_model.to_functional_model()
   functional_model.plot_network()

Functional API
--------------

.. currentmodule:: dlpy.model

The Functional API is a more flexible way of building models. Using the Functional API,
you can easily specify architectures with multi-input and multi-output layers. To use the
Functional API, define the architecture like you would call functions. That is,
pass each input tensor as an argument to a layer to produce an output tensor. Once
you complete the model, pass the inputs and outputs to the :class:`Model` class
and compile it.

.. ipython:: python

   from dlpy.layers import Input, Conv2d, Concat, OutputLayer
   from dlpy import Model
   input1 = Input(n_channels = 1, width = 28, height = 28)
   input2 = Input(n_channels = 3, width = 28, height = 28)
   conv1 = Conv2d(2, name='conv1')(Conv2d(2, name='conv0')(input1))
   conv2 = Conv2d(2, name='conv2')(input1)
   conv3 = Conv2d(2, name='conv3')(input2)
   output2 = OutputLayer(n=2)(conv3)
   concat1 = Concat()([conv1, conv2, conv3])
   output1 = OutputLayer(n=2)(concat1)
   model1 = Model(conn = sess, inputs = [input1, input2], outputs = [output1, output2])
   model1.compile()

Using Pre-built Models
----------------------

DLPy supplies many industry standard models for a variety of computer vision and natural
language processing tasks. These models are generally built with a single function call and
can be customized using Sequential model building methods (including
:func:`Sequential.to_functional_model`). This flexibility enables you to quickly
customize existing models.

For a full list of the pre-build models, see
`Pre-built Models for Computer Vision Tasks <./api.html#pre-built-models-for-computer-vision-tasks>`_
and `Pre-built Models for Natural Language Processing Tasks <./api.html#pre-built-models-for-nlp-tasks>`_.

The following example shows how you can start with a stock ResNet18 model, remove the
default output layer, and add two custom output layers using the Functional API.

.. ipython:: python

   from dlpy.applications import ResNet18_SAS
   resnet = ResNet18_SAS(sess, n_classes=1000, width=224, height=224, n_channels=3)
   resnet_base = resnet.to_functional_model(stop_layers=resnet.layers[-1])

   input_1 = Input(n_channels = 1, width = 28, height = 28)
   resnet_base_tensor = resnet_base(input_1)
   output_0 = OutputLayer(n=2)(resnet_base_tensor)
   output_1 = OutputLayer(n=100)(resnet_base_tensor)
   modified_resnet = Model(sess, inputs=input_1, outputs=[output_0,output_1])
   modified_resnet.compile()
   modified_resnet.plot_network()

Loading Pre-trained Weights
---------------------------

Regardless of whether you build a model from scratch or use a pre-built model, you can
load pre-trained weights as long as the weights were trained from a model with an identical
architecture. The most common way to load weights is to use :func:`Model.load_weights`.

Select pre-trained weights can be downloaded from
`Deep Learning Models and Tools <https://support.sas.com/documentation/prod-p/vdmml/zip/>`_.
