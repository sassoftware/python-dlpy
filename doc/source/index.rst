.. Copyright SAS Institute

:tocdepth: 4

**********************************
SAS Deep Learning Python Interface
**********************************

.. module:: dlpy

**Date**: |today| **Version**: |version|

**Installers:** `<https://github.com/sassoftware/dlpy/releases>`_

**Source Repository:** `<https://github.com/sassoftware/dlpy>`_

**Issues & Ideas:** `<https://github.com/sassoftware/dlpy/issues>`_


DLPy is a high-level package for the Python APIs created for the SAS Viya 3.3 (and newer) Deep Learning and Image action sets. It provides a convenient way to perform deep learning image processing in the SAS CAS environment.

Key Features
============

Keras-like API
--------------

DLPy uses an API similar to `Keras <https://keras.io/>`_ to build the deep learning models. It even allows users to run Keras code with very few modifications to smoothly access SAS analytic and deep learning actions.

Built-in Networks
-----------------

DLPy provides many prebuilt models, including VGG11/13/16/19, ResNet34/50/101/152, wide_resnet, dense_net. The pretrained weights using ImageNet data are also provided for the following models:

   * VGG16
   * VGG19
   * ResNet50
   * ResNet101
   * ResNet152

Those models can be easily used for your own tasks via transfer learning.

Built-in Model Analysis Tools
-----------------------------
Users can conduct a heatmap analysis for deep learning models to reveal the mystery of the black boxes.

Supporting Functions
--------------------
We provide multiple supporting functions for deep learning related tasks, include the visualization of the networks, feature maps and prediction results, data splitting, data augmentation, model deployment, and so on.

Image Data Input, Output and Processing
---------------------------------------
DLPy contains a specific module for image data called ImageTable. It is a subclass of :class:`swat.CASTable` which supports several new features, including reading and writing images from the disc, resizing, cropping and display the images. 

Easy Interaction With Open Source Packages
------------------------------------------
DLPy enables users to fetch the data from CAS session to local client and convert it to the popular data formats such as numpy array or DataFrame. The converted data can be smoothly fed into models in other open source packages such as scikit-learn.


.. toctree::
   :maxdepth: 3

   install
   getting-started
   api
   license


Index
==================

* :ref:`genindex`

