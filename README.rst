***********************************
DLPy - SAS Deep Learning Python API
***********************************

What is DLPy?
=============
DLPy is a high-level package for the Python APIs created for the 
SAS Viya 3.3 (and newer) Deep Learning and Image action sets. 
A SAS Viya VDMML license is required to use these action sets.  DLPy 
provides a convenient way to perform deep learning image processing. 
DLPy uses a familiar Keras-style Python API to access and utilize SAS 
Viya Deep Learning actions in the SAS Cloud Analytic Services (CAS) 
environment. Users who are new to SAS CAS programming, but are 
familiar with other open-source deep learning packages, can use the 
intuitive DLPy interface to run Keras code (with very few modifications) 
to smoothly access SAS analytic and deep learning actions in the 
SAS CAS environment.

The DLPy package is mainly designed for image classification problems 
using Convolutional Neural Network (CNN) models. DLPy currently enables 
GPU support, but on an experimental basis.  Support for Recurrent Neural 
Networks (RNNs) and object detection is under development, and will 
be added to the DLPy package in future releases.


Installing DLPy
===============
SAS provides APIs designed for use with Python 2.7 and Python 3.4+ at
https://github.com/sassoftware/python-dlpy/releases/.

To install DLPy, run the following using the ``pip`` command from your 
Python installation::

    pip install sas-dlpy

Note: To enable graphic visualizations of DLPy deep learning models, 
it is recommended that you download and install the open source graph 
visualization software called Graphviz. Graphviz is available at
https://www.graphviz.org/download/.


Documentation
=============

The API documentation is located at https://sassoftware.github.io/python-dlpy/.


Getting Started with DLPy
=========================
Before you can use the DLPy package, you will need a running SAS CAS 
server and the SWAT (SAS Scripting Wrapper for Analytics Transfer) package. 
SWAT enables you to access and interact with SAS CAS. The SWAT package 
can connect to the binary port or the HTTP port of your CAS host.

In addition to the CAS host and port information, you need a CAS userID 
and password to connect. See your system administrator for details 
if you do not have a CAS account.

To connect to a CAS server, import SWAT and use the swat.CAS class to 
create a connection::

    >>> import swat
    >>> sess = swat.CAS('cloud.example.com', 5570)
	
Next, import the DLPy package, and then build a simple convolutional 
neural network (CNN) model.

Import DLPy model functions::

    >>> from dlpy import Model, Sequential

Import DLPy application functions::

    >>> from dlpy.applications import *
	
Use DLPy to create a sequential model and name it 'Simple_CNN'::

    >>> model1 = Sequential(sess, model_table = 'Simple_CNN')
	
Now define an input layer to add to model1::

    # The input shape contains RGB images (3 channels)
    # The model images are 224 px in height and 224 px in width

    >>> model1.add(InputLayer(3,224,224))

    NOTE: Input layer added.
	
Now, add a 2D convolution layer and a pooling layer::

    # Add 2-Dimensional Convolution Layer to model1
    # that has 8 filters and a kernel size of 7. 

    >>> model1.add(Conv2d(8,7))

    NOTE: Convolutional layer added.

    # Add Pooling Layer of size 2

    >>> model1.add(Pooling(2))

    NOTE: Pooling layer added.
    
Now, add an additional pair of 2D convolution and pooling layers::

    # Add another 2D convolution Layer that has 8 filters
    # and a kernel size of 7 

    >>> model1.add(Conv2d(8,7))

    NOTE: Convolutional layer added.

    # Add a pooling layer of size 2 to # complete the second pair of layers. 
    
    >>> model1.add(Pooling(2))

    NOTE: Pooling layer added.
    
Add a fully connected layer::

    # Add Fully-Connected Layer with 16 units
    
    >>> model1.add(Dense(16))

    NOTE: Fully-connected layer added.
    
Finally, add the output layer::

    # Add an output layer that has 2 nodes and uses
    # the Softmax activation function 
    
    >>> model1.add(OutputLayer(act='softmax',n=2))

    NOTE: Output layer added.
    NOTE: Model compiled successfully 
    
Display a print summary of the table::

    # Display a brief summary table of model1
    
    >>> model1.print_summary()

    *==================*===============*========*============*=================*======================*    
    |   Layer (Type)   |  Kernel Size  | Stride | Activation |   Output Size   | Number of Parameters |    
    *------------------*---------------*--------*------------*-----------------*----------------------*    
    | Data(Input)      |     None      |  None  |    None    |  (224, 224, 3)  |        0 / 0         |    
    | Conv1_1(Convo.)  |    (7, 7)     |   1    |    Relu    |  (224, 224, 8)  |       1176 / 8       |    
    | Pool1(Pool)      |    (2, 2)     |   2    |    Max     |  (112, 112, 8)  |        0 / 0         |    
    | Conv2_1(Convo.)  |    (7, 7)     |   1    |    Relu    |  (112, 112, 8)  |       3136 / 8       |    
    | Pool2(Pool)      |    (2, 2)     |   2    |    Max     |   (56, 56, 8)   |        0 / 0         |    
    | FC1(F.C.)        |  (25088, 16)  |  None  |    Relu    |       16        |     401408 / 16      |    
    | Output(Output)   |    (16, 2)    |  None  |  Softmax   |        2        |        32 / 2        |    
    *==================*===============*========*============*=================*======================*    
    |Total Number of Parameters: 405,786                                                              |    
    *=================================================================================================*
    
    # Use Graphviz to display model network
    
    >>> model1.plot_network()
    
    <graphviz.dot.Digraph at 0x28d5cee32b0>
    
.. image:: https://raw.githubusercontent.com/sassoftware/python-dlpy/master/doc/images/model1_network.png

    
Resources
=========

`SAS SWAT for Python <http://github.com/sassoftware/python-swat/>`_

`Python <http://www.python.org/>`_
