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
   :suppress:

   import os 
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = None
   password = None

.. ipython:: python

   import swat
   sess = swat.CAS(host, port, userid, password)

Now that we can import the DLPy package and build a simple CNN model:

.. ipython:: python

   from dlpy import Model, Sequential
   from dlpy.layers import * 
   from dlpy.applications import *
   
.. ipython:: python

   model1 = Sequential(sess, model_table='Simple_CNN')

.. ipython:: python

   model1.add(InputLayer(3, 224, 224))
   model1.add(Conv2d(8, 7))
   model1.add(Pooling(2))
   model1.add(Conv2d(8, 7))
   model1.add(Pooling(2))
   model1.add(Dense(16))
   model1.add(OutputLayer(act='softmax', n=2))

.. ipython:: python

   model1.print_summary()

.. ipython:: python

   # You need install graphviz to run this, otherwise, just skip this line.
   model1.plot_network()


.. ipython:: python
   :suppress:

   sess.endsession()
