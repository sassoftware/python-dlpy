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

from dlpy.model import Model
from dlpy.keras_models import model_inceptionv3
from dlpy.sequential import Sequential
from dlpy.layers import (Conv2d, BN, Pooling, Concat, OutputLayer, InputLayer)
from .application_utils import get_layer_options, input_layer_options


def InceptionV3(conn, model_table='InceptionV3',
                n_classes=1000, n_channels=3, width=299, height=299, scale=1,
                random_flip=None, random_crop=None, offsets=(103.939, 116.779, 123.68),
                pre_trained_weights=False, pre_trained_weights_file=None, include_top=False,
                random_mutation=None):
    '''
    Generates a deep learning model with the Inceptionv3 architecture with batch normalization layers.

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_table : string, optional
        Specifies the name of CAS table to store the model in.
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 1000
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 299
    height : int, optional
        Specifies the height of the input layer.
        Default: 299
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0
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
        Specifies whether to use the pre-trained weights from ImageNet data set
        Default: False
    pre_trained_weights_file : string, optional
        Specifies the file name for the pretained weights.
        Must be a fully qualified file name of SAS-compatible file (*.caffemodel.h5)
        Note: Required when pre_train_weight=True.
    include_top : bool, optional
        Specifies whether to include pre-trained weights of the top layers,
        i.e. the FC layers
        Default: False
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'

    Returns
    -------
    :class:`Sequential`
        If `pre_train_weight` is `False`
    :class:`Model`
        If `pre_train_weight` is `True`

    References
    ----------
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf

    '''

    conn.retrieve('loadactionset', _messagelevel='error', actionset='deeplearn')

    # get all the parms passed in
    parameters = locals()

    if not pre_trained_weights:
        model = Sequential(conn=conn, model_table=model_table)

        # get the input parameters
        input_parameters = get_layer_options(input_layer_options, parameters)
        model.add(InputLayer(**input_parameters))

        # 299 x 299 x 3
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 149 x 149 x 32
        model.add(Conv2d(n_filters=32, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 147 x 147 x 32
        model.add(Conv2d(n_filters=64, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        # 147 x 147 x 64
        model.add(Pooling(width=3, height=3, stride=2, pool='max', padding=0))

        # 73 x 73 x 64
        model.add(Conv2d(n_filters=80, width=1, height=1, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 73 x 73 x 80
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=1,
                         act='identity', include_bias=False, padding=0))
        model.add(BN(act='relu'))
        # 71 x 71 x 192
        pool2 = Pooling(width=3, height=3, stride=2, pool='max', padding=0)
        model.add(pool2)


        # mixed 0: output 35 x 35 x 256

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[pool2]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[pool2]))
        model.add(Conv2d(n_filters=32, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed0 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 1: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed1 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 2: output 35 x 35 x 288

        # branch1x1
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch5x5
        model.add(Conv2d(n_filters=48, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=64, width=5, height=5, stride=1,
                         act='identity', include_bias=False))
        branch5x5 = BN(act='relu')
        model.add(branch5x5)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed2 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch5x5, branch3x3dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 3: output 17 x 17 x 768

        # branch3x3
        model.add(Conv2d(n_filters=384, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0,
                         src_layers=[concat]))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch3x3dbl
        model.add(Conv2d(n_filters=64, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=96, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3dbl = BN(act='relu')
        model.add(branch3x3dbl)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed3 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch3x3dbl, branch_pool])
        model.add(concat)


        # mixed 4: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=128, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=128, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed4 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 5, 6: output 17 x 17 x 768
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch7x7
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            branch7x7 = BN(act='relu')
            model.add(branch7x7)

            # branch7x7dbl
            model.add(Conv2d(n_filters=160, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=160, width=1, height=7, stride=1,
                             act='identity', include_bias=False))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                             act='identity', include_bias=False))
            branch7x7dbl = BN(act='relu')
            model.add(branch7x7dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                        branch_pool])
            model.add(concat)


        # mixed 7: output 17 x 17 x 768

        # branch1x1
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        branch1x1 = BN(act='relu')
        model.add(branch1x1)

        # branch7x7
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        branch7x7 = BN(act='relu')
        model.add(branch7x7)

        # branch7x7dbl
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        branch7x7dbl = BN(act='relu')
        model.add(branch7x7dbl)

        # branch_pool
        model.add(Pooling(width=3, height=3, stride=1, pool='average',
                          src_layers=[concat]))
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False))
        branch_pool = BN(act='relu')
        model.add(branch_pool)

        # mixed7 concat
        concat = Concat(act='identity',
                        src_layers=[branch1x1, branch7x7, branch7x7dbl,
                                    branch_pool])
        model.add(concat)


        # mixed 8: output 8 x 8 x 1280

        # branch3x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=320, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch3x3 = BN(act='relu')
        model.add(branch3x3)

        # branch7x7x3
        model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                         act='identity', include_bias=False,
                         src_layers=[concat]))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=7, height=1, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=1, height=7, stride=1,
                         act='identity', include_bias=False))
        model.add(BN(act='relu'))
        model.add(Conv2d(n_filters=192, width=3, height=3, stride=2,
                         act='identity', include_bias=False, padding=0))
        branch7x7x3 = BN(act='relu')
        model.add(branch7x7x3)

        # branch_pool
        branch_pool = Pooling(width=3, height=3, stride=2, pool='max',
                              padding=0, src_layers=[concat])
        model.add(branch_pool)

        # mixed8 concat
        concat = Concat(act='identity',
                        src_layers=[branch3x3, branch7x7x3, branch_pool])
        model.add(concat)


        # mixed 9, 10:  output 8 x 8 x 2048
        for i in range(2):
            # branch1x1
            model.add(Conv2d(n_filters=320, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch1x1 = BN(act='relu')
            model.add(branch1x1)

            # branch3x3
            model.add(Conv2d(n_filters=384, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            branch3x3 = BN(act='relu')
            model.add(branch3x3)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_1 = BN(act='relu')
            model.add(branch3x3_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3]))
            branch3x3_2 = BN(act='relu')
            model.add(branch3x3_2)

            branch3x3 = Concat(act='identity',
                               src_layers=[branch3x3_1, branch3x3_2])
            model.add(branch3x3)

            # branch3x3dbl
            model.add(Conv2d(n_filters=448, width=1, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[concat]))
            model.add(BN(act='relu'))
            model.add(Conv2d(n_filters=384, width=3, height=3, stride=1,
                             act='identity', include_bias=False))
            branch3x3dbl = BN(act='relu')
            model.add(branch3x3dbl)

            model.add(Conv2d(n_filters=384, width=3, height=1, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_1 = BN(act='relu')
            model.add(branch3x3dbl_1)

            model.add(Conv2d(n_filters=384, width=1, height=3, stride=1,
                             act='identity', include_bias=False,
                             src_layers=[branch3x3dbl]))
            branch3x3dbl_2 = BN(act='relu')
            model.add(branch3x3dbl_2)

            branch3x3dbl = Concat(act='identity',
                                  src_layers=[branch3x3dbl_1, branch3x3dbl_2])
            model.add(branch3x3dbl)

            # branch_pool
            model.add(Pooling(width=3, height=3, stride=1, pool='average',
                              src_layers=[concat]))
            model.add(Conv2d(n_filters=192, width=1, height=1, stride=1,
                             act='identity', include_bias=False))
            branch_pool = BN(act='relu')
            model.add(branch_pool)

            # concat
            concat = Concat(act='identity',
                            src_layers=[branch1x1, branch3x3,
                                        branch3x3dbl, branch_pool])
            model.add(concat)


        # calculate dimensions for global average pooling
        w = max((width - 75) // 32 + 1, 1)
        h = max((height - 75) // 32 + 1, 1)

        # global average pooling
        model.add(Pooling(width=w, height=h, stride=1, pool='average',
                          padding=0, src_layers=[concat]))

        # output layer
        model.add(OutputLayer(n=n_classes))

        return model

    else:
        if pre_trained_weights_file is None:
            raise ValueError('\nThe pre-trained weights file is not specified.\n'
                             'Please follow the steps below to attach the '
                             'pre-trained weights:\n'
                             '1. Go to the website '
                             'https://support.sas.com/documentation/prod-p/vdmml/zip/ '
                             'and download the associated weight file.\n'
                             '2. Upload the *.h5 file to '
                             'a server side directory which the CAS '
                             'session has access to.\n'
                             '3. Specify the pre_train_weight_file using '
                             'the fully qualified server side path.')
        print('NOTE: Scale is set to 1/127.5, and offsets 1 to '
              'match Keras preprocessing.')
        model_cas = model_inceptionv3.InceptionV3_Model(
            s=conn, model_table=model_table, n_channels=n_channels,
            width=width, height=height, random_crop=random_crop,
            offsets=[1, 1, 1], random_flip=random_flip, random_mutation=random_mutation)

        if include_top:
            if n_classes != 1000:
                warnings.warn('If include_top = True, '
                              'n_classes will be set to 1000.', RuntimeWarning)
            model = Model.from_table(model_cas)
            model.load_weights(path=pre_trained_weights_file, labels=True)
            return model

        else:
            model = Model.from_table(model_cas, display_note=False)
            model.load_weights(path=pre_trained_weights_file)

            weight_table_options = model.model_weights.to_table_params()
            weight_table_options.update(dict(where='_LayerID_<218'))
            model._retrieve_('table.partition', table=weight_table_options,
                             casout=dict(replace=True,
                                         **model.model_weights.to_table_params()))
            model._retrieve_('deeplearn.removelayer', model=model_table,
                             name='predictions')
            model._retrieve_('deeplearn.addlayer', model=model_table,
                             name='predictions',
                             layer=dict(type='output', n=n_classes, act='softmax'),
                             srcLayers=['avg_pool'])
            model = Model.from_table(conn.CASTable(model_table))

            return model
