**********************************
DLPy: SAS Deep Learning Python API
**********************************

What is DLPy?
=============
DLPy is a high-level package for the Python APIs created for the SAS Viya 3.3 (and newer) Deep Learning and Image action sets. DLPy provides a convenient way to perform deep learning image processing. DLPy uses a familiar Keras-style Python API to access and utilize SAS Viya Deep Learning actions in the SAS Cloud Analytic
Services (CAS) environment. Users who are new to SAS CAS programming, but are familiar with other opensource
deep learning packages, can use the intuitive DLPy interface to run Keras code (with very few
modifications) to smoothly access SAS analytic and deep learning actions in the SAS CAS environment.

The new DLPy package is mainly designed for image classification problems using Convolutional Neural
Network (CNN) models. DLPy currently enables GPU support, but on an experimental basis. New support for
Recurrent Neural Networks (RNNs) and object detection is under development, and will be added to the DLPy
package in future releases.


Installing DLPy
===============
SAS provides APIs designed for use with Python 2.7 and Python 3.4. Both are available on the GitHub repository here: https://github.com/sassoftware/dlpy/releases.

Note: To enable graphic visualizations of DLPy deep learning models, it is recommended that you download and install the open source graph visualization software called Graphviz. Graphviz is available here: https://www.graphviz.org/download/

The Graphviz Python package has very few dependencies. All of the dependencies are included in Anaconda Python.

To install DLPy, open an operating system console, navigate to the folder location where you downloaded DLPy, and submit the following:

    pip install dlpy


Resources
=========

`SAS SWAT for Python <http://github.com/sassoftware/python-swat/>`_

`Python <http://www.python.org/>`_
