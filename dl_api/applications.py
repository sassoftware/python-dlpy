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
from .ResNet import ResBlockBN, ResBlock


def VGG11(conn, model_name=None,
          n_channels=3, width=224, height=224, n_classes=None, scale=1,
          random_flip='HV', random_crop='unique', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with VGG11 architecture.

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
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using VGG11 architecture.

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
      Function to generate a deep learning model with VGG13 architecture.

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
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using VGG13 architecture.

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
    n_classes : int, optional.
        Specifies the number of classes. If None is assigned, the model will automatically detect the number of
        classes based on the training set.
        Default: None
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
      Function to generate a deep learning model with VGG19 architecture.

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
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using VGG19 architecture.

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


def LeNet5(conn, model_name=None,
           n_channels=1, width=28, height=28, n_classes=None, scale=1,
           random_flip='NONE', random_crop='NONE', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with LeNet5 architecture.

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
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using LeNet5 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))

    model.add(Conv2d(n_filters=6, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=16, width=5, height=5, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=120))
    model.add(Dense(n=84))
    model.add(OutputLayer(n=n_classes))

    return model


def ResNet18(conn, model_name=None, batch_norm_first=True, n_classes=None,
             n_channels=3, width=224, height=224, scale=1,
             random_flip='H', random_crop='UNIQUE', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with ResNet18 architecture.
    
      Parameters:
    
      ----------
      conn :
          Specifies the connection of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      batch_norm_first: boolean, optional.
          Specifies whether to have batch normalization layer before the convolution layer in the residual block.
          For a detailed discussion about this, please refer to this paper:
          He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
          Default: True.
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using ResNet18 architecture.
    
      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [2, 2, 2, 2]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet34(conn, model_name=None, batch_norm_first=True, n_classes=None,
             n_channels=3, width=224, height=224, scale=1,
             random_flip='H', random_crop='UNIQUE', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with ResNet34 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connection of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      batch_norm_first: boolean, optional.
          Specifies whether to have batch normalization layer before the convolution layer in the residual block.
          For a detailed discussion about this, please refer to this paper:
          He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
          Default: True.
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using ResNet34 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
    n_filters_list = [(64, 64), (128, 128), (256, 256), (512, 512)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet50(conn, model_name=None, batch_norm_first=True, n_classes=None,
             n_channels=3, width=224, height=224, scale=1,
             random_flip='H', random_crop='UNIQUE', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with ResNet50 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connection of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      batch_norm_first: boolean, optional.
          Specifies whether to have batch normalization layer before the convolution layer in the residual block.
          For a detailed discussion about this, please refer to this paper:
          He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
          Default: True.
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using ResNet50 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 6, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet101(conn, model_name=None, batch_norm_first=True, n_classes=None,
              n_channels=3, width=224, height=224, scale=1,
              random_flip='H', random_crop='UNIQUE', offsets=(85, 111, 139)):
    '''
      Function to generate a deep learning model with ResNet101 architecture.

      Parameters:

      ----------
      conn :
          Specifies the connection of the CAS connection.
      model_name : string
          Specifies the name of CAS table to store the model.
      batch_norm_first: boolean, optional.
          Specifies whether to have batch normalization layer before the convolution layer in the residual block.
          For a detailed discussion about this, please refer to this paper:
          He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
          Default: True.
      n_classes : int, optional.
          Specifies the number of classes. If None is assigned, the model will automatically detect the number of
          classes based on the training set.
          Default: None
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
      A model object using ResNet101 architecture.

      '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 4, 23, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2, height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def ResNet152(conn, model_name=None, batch_norm_first=True, n_classes=None,
              n_channels=3, width=224, height=224, scale=1,
              random_flip='H', random_crop='UNIQUE', offsets=(85, 111, 139)):
    '''
    Function to generate a deep learning model with ResNet152 architecture.

    Parameters:

    ----------
    conn :
        Specifies the connection of the CAS connection.
    model_name : string
        Specifies the name of CAS table to store the model.
    batch_norm_first: boolean, optional.
        Specifies whether to have batch normalization layer before the convolution layer in the residual block.
        For a detailed discussion about this, please refer to this paper:
        He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
        Default: True.
    n_classes : int, optional.
        Specifies the number of classes. If None is assigned, the model will automatically detect the number of
        classes based on the training set.
        Default: None
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
    A model object using ResNet152 architecture.

    '''

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(64, 7, act='identity', includeBias=False, stride=2))
    model.add(BN(act='relu'))
    model.add(Pooling(width=3, stride=2))

    kernel_sizes_list = [(1, 3, 1)] * 4
    n_filters_list = [(64, 64, 256), (128, 128, 512), (256, 256, 1024), (512, 512, 2048)]
    rep_nums_list = [3, 8, 36, 3]

    for i in range(4):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))

    # Bottom Layers
    pooling_size = (width // 2 // 2 // 2 // 2 // 2,
                    height // 2 // 2 // 2 // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model


def wide_resnet(conn, model_name=None, batch_norm_first=True, depth=2, k=4, n_classes=None,
                n_channels=3, width=32, height=32, scale=1,
                random_flip='H', random_crop='NONE', offsets=(114, 122, 125)):
    '''
    Function to generate a deep learning model with ResNet152 architecture.

    Parameters:

    ----------
    conn :
        Specifies the connection of the CAS connection.
    model_name : string
        Specifies the name of CAS table to store the model.
    batch_norm_first: boolean, optional.
        Specifies whether to have batch normalization layer before the convolution layer in the residual block.
        For a detailed discussion about this, please refer to this paper:
        He, Kaiming, et al. "Identity mappings in deep residual networks." European Conference on Computer Vision. Springer International Publishing, 2016.
        Default: True.
    depth : Int
        Specifies the number of convolution layers added into the model.
        Default : 2
    k : Int
        Specifies the widening factor.
        Default : 4
    n_classes : int, optional.
        Specifies the number of classes. If None is assigned, the model will automatically detect the number of
        classes based on the training set.
        Default: None
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
    A model object using ResNet152 architecture.

    '''

    n_stack = int((depth - 2) / 6)
    in_filters = 16

    model = Sequential(conn=conn, model_name=model_name)

    model.add(InputLayer(n_channels=n_channels, width=width, height=height,
                         scale=scale, offsets=offsets, random_flip=random_flip,
                         random_crop=random_crop))
    # Top layers
    model.add(Conv2d(in_filters, 3, act='identity', includeBias=False, stride=1))
    model.add(BN(act='relu'))

    n_filters_list = [(16 * k, 16 * k), (32 * k, 32 * k), (64 * k, 64 * k)]
    kernel_sizes_list = [(3, 3)] * len(n_filters_list)
    rep_nums_list = [n_stack, n_stack, n_stack]

    for i in range(len(n_filters_list)):
        kernel_sizes = kernel_sizes_list[i]
        n_filters = n_filters_list[i]
        for rep_num in range(rep_nums_list[i]):
            if i == 0:
                strides = 1
            else:
                if rep_num == 0:
                    strides = 2
                else:
                    strides = 1

            model.add(ResBlockBN(kernel_sizes=kernel_sizes, n_filters=n_filters,
                                 strides=strides, batch_norm_first=batch_norm_first))
    model.add(BN(act='relu'))
    # Bottom Layers
    pooling_size = (width // 2 // 2, height // 2 // 2)
    model.add(Pooling(width=pooling_size[0], height=pooling_size[1], pool='mean'))

    model.add(OutputLayer(act='softmax', n=n_classes))

    return model
