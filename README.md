# DLPy - SAS Deep Learning Python API

## What is DLPy?

DLPy is a high-level Python library for the SAS Deep learning features available in SAS Viya.
DLPy is designed to provide an efficient way to apply deep learning methods to image, text, and audio data.
DLPy APIs created following the Keras (https://keras.io/) APIs with a touch of PyTorch (https://pytorch.org/) flavor.

Note that DLPy requires a SAS Viya on the back-end 
(particularly [Visual Data Mining and Machine 
Learning](https://www.sas.com/en_us/software/visual-data-mining-machine-learning.html) 
product).

## Installing DLPy
DLPy versions >= 1.0 works with Python 3+ and Viya 3.4

To install DLPy using ``pip``, run the following::

    pip install sas-dlpy

To install DLPy using ``conda``, run the following::

    conda install -c sas-institute sas-dlpy

**Note:** To enable graphic visualizations of the DLPy deep learning models,
it is recommended that you download and install the open source graph 
visualization software called Graphviz. Graphviz is available at
https://www.graphviz.org/download/.


## Documentation

The API documentation is located at 
[sassoftware.github.io/python-dlpy](https://sassoftware.github.io/python-dlpy/).


## What's new with DLPy 1.0

- Text, audio, and time series support in addition to image
- New APIs for:
  - RNN based tasks: text classification, text generation, and sequence labeling
  - Object detection
  - Time series processing and modeling
  - Processing audio files and creating speech recognition models
- Additional pre-defined network architectures such as DenseNet, DarkNet, Inception, and Yolo
- Enhanced data visualization and metadata handling

Note that with DLPy 1.0, we moved to python-style functions and parameters. 
This might break your old code with camelCase parameters and functions.


# Getting Started with DLPy

Before you can use the DLPy package, you will need a running SAS Viya
server and the SWAT (SAS Scripting Wrapper for Analytics Transfer) package. 
SWAT enables you to access and interact with SAS Viya. The SWAT package
can connect to the binary port or the HTTP port of your SAS Viya host.

In addition to the SAS Viya host and port information, you need a SAS Viya userID
and password to connect. See your system administrator for details 
if you do not have a SAS Viya account.

To connect to a SAS Viya server, import SWAT and use the swat.CAS class to
create a connection::

    >>> import swat
    >>> sess = swat.CAS('cloud.example.com', 5570)

Next, import the DLPy package, and then build a simple convolutional 
neural network (CNN) model.

Import DLPy model functions::

    >>> from dlpy import Model, Sequential
    >>> from dlpy.layers import *

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


**Check out** https://github.com/sassoftware/python-dlpy/tree/master/examples **for more examples.**

## Resources

[SAS SWAT for Python](http://github.com/sassoftware/python-swat/)

[SAS ESPPy](https://github.com/sassoftware/python-esppy)

[SAS Viya](http://www.sas.com/en_us/software/viya.html)

[Python](http://www.python.org/)

Copyright SAS Institute
