.. Copyright SAS Institute

.. currentmodule:: dlpy
.. _api:

*************
API Reference
*************

.. _api.functions:


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

Audio
-----

.. currentmodule:: dlpy.audio

.. autosummary::
   :toctree: generated/
       
   AudioTable
   AudioTable.create_audio_table
   
Timeseries
----------

.. currentmodule:: dlpy.timeseries

.. autosummary::
   :toctree: generated/
   
   TimeseriesTable
   TimeseriesTable.from_table
   TimeseriesTable.from_pandas
   TimeseriesTable.from_localfile
   TimeseriesTable.from_serverfile
   TimeseriesTable.timeseries_formatting
   TimeseriesTable.timeseries_accumlation
   TimeseriesTable.prepare_subsequences
   TimeseriesTable.timeseries_partition
   plot_timeseries   
   
Layers
------

.. currentmodule:: dlpy.layers

.. autosummary::
   :toctree: generated/

   Layer

.. autosummary::
   :toctree: generated/

   InputLayer
   Conv2d
   Pooling
   Dense
   Recurrent
   BN
   Res
   Concat
   Proj
   OutputLayer
   Keypoints
   Detection
   Scale
   Reshape
   


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
   Model.evaluate
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

   DataSpec
   DataSpecNumNomOpts
   Sequence
   TextParms
   Optimizer
   NatGradSolver
   LBFGSolver
   AdamSolver
   MomentumSolver
   VanillaSolver
   Solver
   Gpu


Feature Maps
------------

.. currentmodule:: dlpy.model

.. autosummary::
   :toctree: generated/

   FeatureMaps
   FeatureMaps.display


Sequential Model
----------------

.. currentmodule:: dlpy.sequential

.. autosummary::
   :toctree: generated/

   Sequential
   Sequential.add
   Sequential.pop
   Sequential.switch
   Sequential.compile

   
Residual Networks
-----------------

.. currentmodule:: dlpy.blocks

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
   
   
Pre-Built Models for Computer Vision Tasks
------------------------------------------

.. currentmodule:: dlpy.applications

.. autosummary::
   :toctree: generated/

   LeNet5
   VGG11
   VGG13
   VGG16
   VGG19
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
   ResNet_Wide
   DenseNet
   DenseNet121
   Darknet
   Darknet_Reference
   InceptionV3
   YoloV2
   YoloV2_MultiSize
   Tiny_YoloV1
   Tiny_YoloV2

  
Pre-Built Models for NLP Tasks
------------------------------

.. currentmodule:: dlpy.applications

.. autosummary::
   :toctree: generated/
   
   TextClassification
   TextGeneration
   SequenceLabeling
   SpeechRecognition


Splitting Utilities
-------------------

.. currentmodule:: dlpy.splitting

.. autosummary::
   :toctree: generated/

   two_way_split
   three_way_split
