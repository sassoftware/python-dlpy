.. Copyright SAS Institute

.. currentmodule:: dlpy
.. _api:

*************
API Reference
*************

.. _api.functions:


Pre-built models
----------------

.. currentmodule:: dlpy.applications

.. autosummary::
   :toctree: generated/

   LeNet5
   LeNet5_bn
   VGG11
   VGG11_bn
   VGG13
   VGG13_bn
   VGG16
   VGG16_bn
   VGG19
   VGG19_bn
   ResNet18_SAS
   ResNet18_Caffe
   ResNet34_SAS
   ResNet34_Caffe
   ResNet50_SAS
   ResNet50_Caffe
   ResNet101_SAS
   ResNet101_Caffe
   ResNet152_SAS
   ResNet152_Caffe
   wide_resnet
   DenseNet_Cifar


ImageTable
----------

The :class:`ImageTable` class is a specialized version of :class:`swat.CASTable`
that includes extra methods for working with images.

.. currentmodule:: dlpy.images

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImageTable
   ImageTable.from_table
   ImageTable.load_files


Saving
~~~~~~

.. autosummary::
   :toctree: generated/

   ImageTable.to_files
   ImageTable.to_sashdat


Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImageTable.copy_table
   ImageTable.show


Image Processing
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   ImageTable.crop
   ImageTable.resize
   ImageTable.as_patches
   ImageTable.as_random_patches


Layers
------

.. currentmodule:: dlpy.layers

.. autosummary::
   :toctree: generated/

   Layer
   Layer.to_model_params

.. autosummary::
   :toctree: generated/

   InputLayer
   Conv2d
   Pooling
   Dense
   Recurrent
   BN
   Res
   Proj
   OutputLayer


Models
------

.. currentmodule:: dlpy.model

.. autosummary::
   :toctree: generated/

   Model
   Model.from_table
   Model.from_sashdat
   Model.load
   Model.set_weights
   Model.load_weights
   Model.load_weights_from_caffe
   Model.load_weights_from_keras
   Model.load_weights_from_table
   Model.set_weights_attr
   Model.load_weights_attr
   Model.fit
   Model.tune
   Model.plot_training_history
   Model.predict
   Model.plot_predict_res
   Model.get_feature_maps
   Model.get_features
   Model.heat_map_analysis
   Model.plot_heat_map
   Model.save_to_astore
   Model.save_to_table
   Model.deploy
   Model.count_params
   Model.print_summary
   Model.plot_network


Feature Maps
------------

.. currentmodule:: dlpy.model

.. autosummary::
   :toctree: generated/

   FeatureMaps
   FeatureMaps.display


Functions
---------

.. currentmodule:: dlpy.model

.. autosummary::
   :toctree: generated/

   get_num_configs
   get_str_configs
   extract_input_layer
   extract_conv_layer
   extract_pooling_layer
   extract_batchnorm_layer
   extract_residual_layer
   extract_fc_layer
   extract_output_layer
   layer_to_node
   layer_to_edge
   model_to_graph


Residual Networks
-----------------

.. currentmodule:: dlpy.blocks

.. autosummary::
   :toctree: generated/

   Block

.. autosummary::
   :toctree: generated/

   ResBlock
   ResBlock.compile

.. autosummary::
   :toctree: generated/

   ResBlockBN
   ResBlockBN.compile

.. autosummary::
   :toctree: generated/

   ResBlock_Caffe
   ResBlock_Caffe.compile



Sequential Model
----------------

.. currentmodule:: dlpy.Sequential

.. autosummary::
   :toctree: generated/

   Sequential
   Sequential.add
   Sequential.pop
   Sequential.switch
   Sequential.compile


Splitting Utilities
-------------------

.. currentmodule:: dlpy.splitting

.. autosummary::
   :toctree: generated/

   two_way_split
   three_way_split

