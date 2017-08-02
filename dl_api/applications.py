#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from .layers import *
from .Sequential import Sequential


def VGG11(conn, model_name=None,
          nChannels=3, width=224, height=224, nclass=None, scale=1,
          randomFlip='HV', randomCrop='unique', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with VGG16 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connion of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      pre_train_weight : boolean, optional.
          Specifies whether to use the pre-trained weights from ImageNet data set.
          Default : False.
      include_top : boolean, optional.
          Specifies whether to include pre-trained weights of the top layers, i.e. the FC layers.
          Default : False.
      nchannels : double, optional.
          Specifies the number of the channels of the input layer.
          Default : 3.
      width : double, optional.
          Specifies the width of the input layer.
          Default : 224.
      height : double, optional.
          Specifies the height of the input layer.
          Default : 224.
      scale : double, optional.
          Specifies a scaling factor to apply to each image..
          Default : 1.
      randomFlip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      randomcrop : string, "NONE" or "UNIQUE"
          Specifies how to crop the data in the input layer when image data is used. Images are cropped to the values that
           are specified in the width and height parameters. Only the images with one or both dimensions that are larger
           than those sizes are cropped.
          Default	: "UNIQUE"
      offsets=(double-1 <, double-2, ...>), optional
          Specifies an offset for each channel in the input data. The final input data is set after applying scaling and
          subtracting the specified offsets.
      Default : (85, 111, 139)

      Returns
      -------
      A model object using VGG16 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(nChannels=nChannels, width=width, height=height,
                         scale=scale, offsets=offsets, randomFlip=randomFlip,
                         randomCrop=randomCrop))

    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=nclass))


def VGG13(conn, model_name=None,
          nChannels=3, width=224, height=224, nclass=None, scale=1,
          randomFlip='HV', randomCrop='unique', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with VGG16 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connion of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      pre_train_weight : boolean, optional.
          Specifies whether to use the pre-trained weights from ImageNet data set.
          Default : False.
      include_top : boolean, optional.
          Specifies whether to include pre-trained weights of the top layers, i.e. the FC layers.
          Default : False.
      nchannels : double, optional.
          Specifies the number of the channels of the input layer.
          Default : 3.
      width : double, optional.
          Specifies the width of the input layer.
          Default : 224.
      height : double, optional.
          Specifies the height of the input layer.
          Default : 224.
      scale : double, optional.
          Specifies a scaling factor to apply to each image..
          Default : 1.
      randomFlip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      randomcrop : string, "NONE" or "UNIQUE"
          Specifies how to crop the data in the input layer when image data is used. Images are cropped to the values that
           are specified in the width and height parameters. Only the images with one or both dimensions that are larger
           than those sizes are cropped.
          Default	: "UNIQUE"
      offsets=(double-1 <, double-2, ...>), optional
          Specifies an offset for each channel in the input data. The final input data is set after applying scaling and
          subtracting the specified offsets.
      Default : (85, 111, 139)

      Returns
      -------
      A model object using VGG16 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(nChannels=nChannels, width=width, height=height,
                         scale=scale, offsets=offsets, randomFlip=randomFlip,
                         randomCrop=randomCrop))

    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=nclass))


def VGG16(conn, model_name=None, pre_train_weight=False, include_top=False,
          nChannels=3, width=224, height=224, nclass=None, scale=1,
          randomFlip='HV', randomCrop='unique', offsets=(85, 111, 139)):
    '''
    Function to generate a deep learning model with VGG16 architecture.

    Parameters:

    ----------
    conn :
        Specifies the connection of the CAS connection.
    model_name : string
        Specifies the name of CAS table to store the model.
    pre_train_weight : boolean, optional.
        Specifies whether to use the pre-trained weights from ImageNet data set.
        Default : False.
    include_top : boolean, optional.
        Specifies whether to include pre-trained weights of the top layers, i.e. the FC layers.
        Default : False.
    nchannels : double, optional.
        Specifies the number of the channels of the input layer.
        Default : 3.
    width : double, optional.
        Specifies the width of the input layer.
        Default : 224.
    height : double, optional.
        Specifies the height of the input layer.
        Default : 224.
    scale : double, optional.
        Specifies a scaling factor to apply to each image..
        Default : 1.
    randomFlip : string, "H" | "HV" | "NONE" | "V"
        Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
        is subject to flipping.
        Default	: "HV"
    randomcrop : string, "NONE" or "UNIQUE"
        Specifies how to crop the data in the input layer when image data is used. Images are cropped to the values that
         are specified in the width and height parameters. Only the images with one or both dimensions that are larger
         than those sizes are cropped.
        Default	: "UNIQUE"
    offsets=(double-1 <, double-2, ...>), optional
        Specifies an offset for each channel in the input data. The final input data is set after applying scaling and
        subtracting the specified offsets.
    Default : (85, 111, 139)

    Returns
    -------
    A model object using VGG16 architecture.

    '''

    if include_top:
        nclass = 1000

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(nChannels=nChannels, width=width, height=height,
                         scale=scale, offsets=offsets, randomFlip=randomFlip,
                         randomCrop=randomCrop))

    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=nclass))

    if pre_train_weight:
        conn.table.loadTable(_messagelevel='error',
                             casout=dict(replace=True, name='_imagenet_vgg16_weights_'),
                             caslib='CASTestTmp', path='vgg/vgg16_weights_table.sashdat')
        if include_top:
            model.set_weights('_imagenet_vgg16_weights_')
            conn.table.loadTable(_messagelevel='error',
                                 casout=dict(replace=True, name='_imagenet_vgg16_weights_attr_'),
                       caslib='CASTestTmp', path='vgg16_weights_attrib.sashdat')
            model.set_weights_attr('_imagenet_vgg16_weights_attr_')
        else:
            model.set_weights(weight_tbl=dict(name='_imagenet_vgg16_weights', where='_layerid_<18'))
    return model


def VGG19(conn, model_name=None,
          nChannels=3, width=224, height=224, nclass=None, scale=1,
          randomFlip='HV', randomCrop='unique', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with VGG16 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connion of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      pre_train_weight : boolean, optional.
          Specifies whether to use the pre-trained weights from ImageNet data set.
          Default : False.
      include_top : boolean, optional.
          Specifies whether to include pre-trained weights of the top layers, i.e. the FC layers.
          Default : False.
      nchannels : double, optional.
          Specifies the number of the channels of the input layer.
          Default : 3.
      width : double, optional.
          Specifies the width of the input layer.
          Default : 224.
      height : double, optional.
          Specifies the height of the input layer.
          Default : 224.
      scale : double, optional.
          Specifies a scaling factor to apply to each image..
          Default : 1.
      randomFlip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      randomcrop : string, "NONE" or "UNIQUE"
          Specifies how to crop the data in the input layer when image data is used. Images are cropped to the values that
           are specified in the width and height parameters. Only the images with one or both dimensions that are larger
           than those sizes are cropped.
          Default	: "UNIQUE"
      offsets=(double-1 <, double-2, ...>), optional
          Specifies an offset for each channel in the input data. The final input data is set after applying scaling and
          subtracting the specified offsets.
      Default : (85, 111, 139)

      Returns
      -------
      A model object using VGG16 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(nChannels=nChannels, width=width, height=height,
                         scale=scale, offsets=offsets, randomFlip=randomFlip,
                         randomCrop=randomCrop))

    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Conv2d(nFilters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=nclass))

    return model
