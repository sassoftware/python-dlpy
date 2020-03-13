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
   ImageTable.random_mutations


AudioTable
----------

The :class:`AudioTable` class is a specialized version of :class:`swat.CASTable`
that includes extra methods for working with audio data.

.. currentmodule:: dlpy.audio

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   AudioTable
   AudioTable.create_audio_table
   AudioTable.create_audio_table_speechrecognition
   AudioTable.load_audio_files
  

Audio Processing
~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   AudioTable.extract_audio_features
   AudioTable.load_audio_metadata
   AudioTable.load_audio_metadata_speechrecognition


Timeseries
----------

.. currentmodule:: dlpy.timeseries

.. autosummary::
   :toctree: generated/

   plot_timeseries   

TimeseriesTable
~~~~~~~~~~~~~~~

.. currentmodule:: dlpy.audio

The :class:`TimeseriesTable` class is a specialized version of :class:`swat.CASTable`
that includes extra methods for working with timeseries.

.. currentmodule:: dlpy.timeseries

Constructors
^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/
   
   TimeseriesTable
   TimeseriesTable.from_table
   TimeseriesTable.from_pandas
   TimeseriesTable.from_localfile
   TimeseriesTable.from_serverfile

Utilities
^^^^^^^^^

.. autosummary::
   :toctree: generated/

   TimeseriesTable.timeseries_formatting
   TimeseriesTable.timeseries_accumlation
   TimeseriesTable.prepare_subsequences
   TimeseriesTable.timeseries_partition
   
Layers
------

.. currentmodule:: dlpy.layers

.. autosummary::
   :toctree: generated/

   Layer

.. autosummary::
   :toctree: generated/

   InputLayer
   BN
   ChannelShuffle
   Concat
   Conv2d
   Conv2DTranspose
   Dense
   Detection
   EmbeddingLoss
   FastRCNN
   GlobalAveragePooling2D
   GroupConv2d
   Keypoints
   Pooling
   Proj
   Recurrent
   Res
   Reshape
   ROIPooling
   Scale
   Segmentation
   OutputLayer

Model
-----

The :class:`Model` class is a specialized version of :class:`dlpy.Network`
that adds training, evaluation, tuning, and feature analysis routines.  

.. currentmodule:: dlpy.model

Constructors
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Model
   Model.from_table
   Model.from_keras_model
   Model.from_caffe_model
   Model.from_onnx_model
   Model.from_sashdat
   Model.load


Model Setup
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Model.change_labels
   Model.set_weights
   Model.load_weights
   Model.load_weights_from_caffe
   Model.load_weights_from_keras
   Model.load_weights_from_table
   Model.set_weights_attr
   Model.load_weights_attr


Training
~~~~~~~~

.. autosummary::
   :toctree: generated/

   Model.fit
   Model.tune
   Model.plot_training_history


Inference, Evaluation, and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Model.predict
   Model.forecast
   Model.evaluate
   Model.evaluate_object_detection
   Model.plot_evaluate_res
   Model.get_feature_maps
   Model.get_features
   Model.heat_map_analysis
   Model.plot_heat_map


Saving
~~~~~~

.. autosummary::
   :toctree: generated/

   Model.save_to_astore
   Model.save_to_table
   Model.deploy


Architecture Information
~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Model.count_params
   Model.print_summary
   Model.plot_network
   Model.get_model_info

Solvers
-------

.. autosummary::
   :toctree: generated/

   Solver
   NatGradSolver
   LBFGSolver
   AdamSolver
   MomentumSolver
   VanillaSolver


   
Optimizer
---------

.. autosummary::
   :toctree: generated/

   Optimizer

Learning Rate Scheduler
-----------------------

.. currentmodule:: dlpy.lr_scheduler

.. autosummary::
   :toctree: generated/

   FCMPLR

.. autosummary::
   :toctree: generated/

   FixedLR
   StepLR
   MultiStepLR
   PolynomialLR
   ReduceLROnPlateau
   CyclicLR
   

Parameters
----------

.. currentmodule:: dlpy.model

.. autosummary::
   :toctree: generated/

   DataSpec
   DataSpecNumNomOpts
   Sequence
   TextParms
   Gpu


Metrics
-------

.. currentmodule:: dlpy.metrics

.. autosummary::
   :toctree: generated/
   
   accuracy_score
   confusion_matrix
   plot_roc
   plot_precision_recall
   roc_auc_score
   average_precision_score
   f1_score
   explained_variance_score
   mean_absolute_error
   mean_squared_error
   mean_squared_log_error
   r2_score   


Feature Maps
------------

.. currentmodule:: dlpy.model

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   FeatureMaps

Utilities
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   FeatureMaps.display


Sequential Model
----------------

.. currentmodule:: dlpy.sequential

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Sequential

Utilities
~~~~~~~~~

.. autosummary::
   :toctree: generated/

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
      
.. autosummary::
   :toctree: generated/

   Bidirectional 
   Bidirectional.compile

Pre-Built Models for Computer Vision Tasks
------------------------------------------

.. currentmodule:: dlpy.applications

.. autosummary::
   :toctree: generated/

Image Classification
~~~~~~~~~~~~~~~~~~~~

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
   MobileNetV1
   MobileNetV2
   ShuffleNetV1
   
Object Detection
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   YoloV1
   YoloV2
   YoloV2_MultiSize
   Tiny_YoloV1
   Tiny_YoloV2
   Faster_RCNN
   
Segmentation
~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/
   
   UNet
   
Image captioning
----------------

.. currentmodule:: dlpy.image_captioning

.. autosummary::
   :toctree: generated/
   
   ImageCaptioning
   create_captioning_table
   display_predicted_image_captions
  
Pre-Built Models for NLP Tasks
------------------------------

.. currentmodule:: dlpy.applications

.. autosummary::
   :toctree: generated/
   
   TextClassification
   TextGeneration
   SequenceLabeling
   SpeechRecognition


Speech
------------

.. currentmodule:: dlpy.speech

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   Speech

Utilities
~~~~~~~~~
.. autosummary::
   :toctree: generated/

   Speech.load_acoustic_model
   Speech.load_language_model
   Speech.transcribe
   
   
Speech Utilities
----------------

.. currentmodule:: dlpy.speech_utils

.. autosummary::
   :toctree: generated/

   read_audio
   check_framerate
   check_sampwidth
   convert_framerate
   convert_sampwidth
   calculate_segment_nframes
   segment_audio
   clean_audio
   check_stereo
   convert_stereo_to_mono
   

Splitting Utilities
-------------------

.. currentmodule:: dlpy.splitting

.. autosummary::
   :toctree: generated/

   two_way_split
   three_way_split
