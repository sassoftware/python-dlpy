.. Copyright SAS Institute

:tocdepth: 4

**********************************
SAS Deep Learning Python Interface
**********************************

.. module:: dlpy

**Date**: |today| **Version**: |version|

**Installers:** `<https://github.com/sassoftware/python-dlpy/releases>`_

**Source Repository:** `<https://github.com/sassoftware/python-dlpy>`_

**Issues & Ideas:** `<https://github.com/sassoftware/python-dlpy/issues>`_


DLPy is a high-level package for the Python APIs created for the SAS Viya 3.3 (and newer) Deep Learning back end. It provides a convenient way to apply deep learning functionalities to solve the computer vision, NLP, forecasting, and speech processing problems.

Key Features
============

User friendly API
-----------------

We design DLPy API to be similar to existing packages (e.g., Keras). It is a blend of the familiar easy and lazy Keras flavor and a pinch of PyTorch flavor for more advanced users.

Built-in Networks
-----------------

DLPy provides many prebuilt models, including VGG and ResNet. The pretrained weights using ImageNet data are also provided for those models. This would give you a warm start on your favourite task via transfer learning.

Built-in Model Analysis Tools
-----------------------------
Users can conduct a heatmap analysis for deep learning models to reveal the mystery of the black boxes in the computer vision problems.

Supporting Functions
--------------------
We provide multiple supporting functions for deep learning related tasks, include the visualization of the networks, feature maps and prediction results, data splitting, data augmentation, model deployment, and so on.

Easy Model Creation For Specific Tasks
---------------------------------------
Every deep learning package requires a lot of tricks to define models for different tasks. We are doing our best to remove this burden and therefore we created predefined models for specific tasks that takes care of knowing the magic syntax.

Easy Interaction With Open Source Packages
------------------------------------------
DLPy enables users to fetch the data from the back end (SAS Viya session) to the local client and convert it to a popular data format such as numpy array or DataFrame. The converted data can be smoothly fed into models in other open source packages such as scikit-learn.


.. toctree::
   :maxdepth: 3

   whatsnew
   install
   getting-started
   api
   license


Index
=====

* :ref:`genindex`

