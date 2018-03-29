.. Copyright SAS Institute

.. currentmodule:: dlpy

***************
Getting Started
***************

Before you can use the DLPy package, you will need a running CAS server and the SWAT package.
The SWAT package can connect to either the binary port or the HTTP port.  If you have the 
option of either, the binary protocol will give you better performance. 

Other than the CAS host and port, you just need a user name and password to connect. 
User names and passwords can be implemented in various ways, so you may need to see your
system administrator on how to acquire an account.

To connect to a CAS server, you simply import SWAT and use the :class:`swat.CAS` class
to create a connection.

.. ipython:: python

   import os 
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = None
   password = None

.. ipython:: python

   from swat import * 
   sess = CAS(host, port,userid, password)

   
Next, import the DLPy package, and then build a simple convolutional neural network (CNN) model.

.. ipython:: python
   from dlpy import Model, Sequential

Import DLPy layer functions

.. ipython:: python
   from dlpy.layers import *
   
Use DLPy to create a sequential model and name it 'Simple_CNN' 

.. ipython:: python
   model1 = Sequential(sess, model_table = 'Simple_CNN')
   
Now define an input layer to add to model1

.. ipython:: python
   # The input shape contains RGB images (3 channels)
   # The model images are 224 px in height and 224 px in width
   model1.add(InputLayer(3,224,224))

   
Now, add a 2D convolution layer and a pooling layer.

.. ipython:: python
   # Add 2-Dimensional Convolution Layer to model1
   # that has 8 filters and a kernel size of 7. 

   model1.add(Conv2d(8,7)


.. ipython:: python
   # Add Pooling Layer of size 2
   model1.add(Pooling(2))


   
Now, add an additional pair of 2D convolution and pooling layers.
.. ipython:: python

   # Add another 2D convolution Layer that has 8 filters
   # and a kernel size of 7 
   
   model1.add(Conv2d(8,7)



.. ipython:: python
   # Add a pooling layer of size 2 to # complete the second pair of layers. 
   
   model1.add(Pooling(2))

   
Add a fully connected layer.

.. ipython:: python
   # Add Fully-Connected Layer with 16 units
   
   model1.add(Dense(16))

   
Finally, add the output layer.

.. ipython:: python
   # Add an output layer that has 2 nodes and uses
   # the Softmax activation function 
   
   model1.add(OutputLayer(act='softmax',n=2))

   
Display a print summary of the table.

.. ipython:: python
   #Display a brief summary table of model1
   
   model1.print_summary()


Use the open source utility Graphviz to display a plot of the model network. 
Graphviz is available here: https://www.graphviz.org/download/. 
If you do not have Graphviz, skip this instruction.

.. ipython:: python
   # Use Graphviz to display model network
   
   model1.plot_network()
   
.. ipython:: python
:suppress:

   sess.endsession()