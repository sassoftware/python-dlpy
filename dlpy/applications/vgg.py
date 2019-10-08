#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import warnings

from dlpy.sequential import Sequential
from dlpy.model import Model
from dlpy.layers import InputLayer, Conv2d, BN, Pooling, OutputLayer, Dense, Reshape
from dlpy.utils import DLPyError, check_layer_class
from dlpy.caffe_models import (model_vgg16, model_vgg19)
from .application_utils import get_layer_options, input_layer_options

def VGG11(conn, model_table='VGG11', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
          random_mutation=None):
    '''
    Generates a deep learning model with the VGG11 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final
        input data is set after applying scaling and subtracting the
        specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

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

    return model


def VGG13(conn, model_table='VGG13', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
          random_mutation=None):
    '''
    Generates a deep learning model with the VGG13 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    model = Sequential(conn=conn, model_table=model_table)

    # get the input parameters
    input_parameters = get_layer_options(input_layer_options, parameters)
    model.add(InputLayer(**input_parameters))

    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=64, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=128, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=256, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Conv2d(n_filters=512, width=3, height=3, stride=1, act='identity', include_bias=False))
    model.add(BN(act='relu'))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Dense(n=4096, dropout=0.5))
    model.add(Dense(n=4096, dropout=0.5))

    model.add(OutputLayer(n=n_classes))

    return model


def VGG16(conn, model_table='VGG16', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
          random_mutation=None, reshape_after_input=None):
    '''
    Generates a deep learning model with the VGG16 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers)
        Default: False
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    reshape_after_input : :class:`Reshape`, optional
        Specifies whether to add a reshape layer after the input layer.

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    # check the type
    check_layer_class(reshape_after_input, Reshape)

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

        if reshape_after_input:
            model.add(reshape_after_input)

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

        model.add(Dense(n=4096, dropout=0.5, name='fc6'))
        model.add(Dense(n=4096, dropout=0.5, name='fc7'))

        model.add(OutputLayer(n=n_classes, name='fc8'))

        return model

    else:
        # TODO: I need to re-factor loading / downloading pre-trained models.
        # something like pytorch style

        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg16.VGG16_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets,
                                            random_mutation=random_mutation, reshape_after_input=reshape_after_input)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<19'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model


def VGG19(conn, model_table='VGG19', n_classes=1000, n_channels=3, width=224, height=224, scale=1,
          random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
          pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
          random_mutation=None):
    '''
    Generates a deep learning model with the VGG19 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 224
    height : int, optional
        Specifies the height of the input layer.
        Default: 224
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    random_flip : string, optional
        Specifies how to flip the data in the input layer when image data is
        used. Approximately half of the input data is subject to flipping.
        Valid Values: 'h', 'hv', 'v', 'none'
    random_crop : string, optional
        Specifies how to crop the data in the input layer when image data is
        used. Images are cropped to the values that are specified in the width
        and height parameters. Only the images with one or both dimensions
        that are larger than those sizes are cropped.
        Valid Values: 'none', 'unique', 'randomresized', 'resizethencrop'
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
        Default: (103.939, 116.779, 123.68)
    pre_trained_weights : bool, optional
        Specifies whether to use the pre-trained weights trained on the ImageNet data set.
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pre-trained weights.
        Must be a fully qualified file name of SAS-compatible file (e.g., *.caffemodel.h5)
        Note: Required when pre_trained_weights=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers (i.e., the FC layers).
        Default: False
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`
        If `pre_trained_weights` is False
    :class:`Model`
        If `pre_trained_weights` is True

    References
    ----------
    https://arxiv.org/pdf/1409.1556.pdf

    '''
    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

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

    else:
        if pre_trained_weights_file is None:
            raise DLPyError('\nThe pre-trained weights file is not specified.\n'
                            'Please follow the steps below to attach the pre-trained weights:\n'
                            '1. Go to the website https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                            'and download the associated weight file.\n'
                            '2. Upload the *.h5 file to '
                            'a server side directory which the CAS session has access to.\n'
                            '3. Specify the pre_trained_weights_file using the fully qualified server side path.')

        model_cas = model_vgg19.VGG19_Model(s=conn, model_table=model_table, n_channels=n_channels,
                                            width=width, height=height, random_crop=random_crop, offsets=offsets,
                                            random_flip=random_flip, random_mutation=random_mutation)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, n_classes will be set to 1000.', RuntimeWarning)

            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:

            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<22'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True, **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table, name='fc8')
            model._retrieve_('deeplearn.addlayer', model=model_table, name='fc8',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['fc7'])
            model = Model.from_table(conn.CASTable(model_table))

            return model
