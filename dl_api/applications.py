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
          n_channels=3, width=224, height=224, n_classes=None, scale=1,
          random_flip='HV', random_crop='unique', offsets=(85, 111, 139)):
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
      n_channels : double, optional.
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
      random_flip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      random_crop : string, "NONE" or "UNIQUE"
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

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))


def VGG13(conn, model_name=None,
          n_channels=3, width=224, height=224, n_classes=None, scale=1,
          random_flip='HV', random_crop='unique', offsets=(85, 111, 139)):
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
      n_channels : double, optional.
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
      random_flip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      random_crop : string, "NONE" or "UNIQUE"
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

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))


def VGG16(conn, model_name=None, pre_train_weight=False, include_top=False,
          n_channels=3, width=224, height=224, n_classes=None, scale=1,
          random_flip='HV', random_crop='unique', offsets=(85, 111, 139)):
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
    n_channels : double, optional.
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
    random_flip : string, "H" | "HV" | "NONE" | "V"
        Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
        is subject to flipping.
        Default	: "HV"
    random_crop : string, "NONE" or "UNIQUE"
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
        n_classes = 1000

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    if pre_train_weight:
        from .utils import random_name
        CAS_lib_name = random_name('CASLIB')
        conn.retrieve('addcaslib', _messagelevel='error',
                      activeonadd=False,
                      name=CAS_lib_name,
                      path='/dept/cas/leliuz/DL_MODELS',
                      dataSource=dict(srcType="DNFS"))

        conn.retrieve('table.loadtable', _messagelevel='error',
                      casout=dict(replace=True, name='{}_weights'.format(model_name)),
                      caslib=CAS_lib_name, path='VGG16_WEIGHTS.sashdat')
        if include_top:
            model.set_weights('{}_weights'.format(model_name))
            conn.retrieve('table.loadtable', _messagelevel='error',
                          casout=dict(replace=True, name='{}_weights_attr'.format(model_name)),
                          caslib=CAS_lib_name, path='VGG16_WEIGHTS_ATTR.sashdat')
            model.set_weights_attr('{}_weights_attr'.format(model_name))
        else:
            model.set_weights(weight_tbl=dict(name='{}_weights'.format(model_name), where='_layerid_<18'))

        conn.retrieve('dropcaslib', _messagelevel='error',
                      caslib=CAS_lib_name)
    return model


def VGG19(conn, model_name=None,
          n_channels=3, width=224, height=224, n_classes=None, scale=1,
          random_flip='HV', random_crop='unique', offsets=(85, 111, 139)):
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
      n_channels : double, optional.
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
      random_flip : string, "H" | "HV" | "NONE" | "V"
          Specifies how to flip the data in the input layer when image data is used. Approximately half of the input data
          is subject to flipping.
          Default	: "HV"
      random_crop : string, "NONE" or "UNIQUE"
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

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))
    model.add(OutputLayer(n=n_classes))

    return model
