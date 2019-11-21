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


from dlpy.sequential import Sequential
from dlpy.layers import InputLayer, Conv2d, BN, Pooling, Detection, Dense, Reshape, Concat
from dlpy.utils import DLPyError
from .application_utils import get_layer_options, input_layer_options, not_supported_feature


def YoloV2(conn, anchors, model_table='YoloV2', n_channels=3, width=416, height=416, scale=1.0 / 255,
           random_mutation=None, act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
           num_to_force_coord=None, random_flip=None, random_crop=None):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the input layer.
        Valid Values: 'none', 'random'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information.
        Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    if len(anchors) != 2 * predictions_per_grid:
        raise DLPyError('The size of the anchor list in the detection layer for YOLOv2 should be equal to '
                        'twice the number of predictions_per_grid.')

    model = Sequential(conn=conn, model_table=model_table)

    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)

    if input_parameters['width'] != input_parameters['height']:
        print(not_supported_feature('Non-square yolo model training', 'height=width'))
        input_parameters['height'] = input_parameters['width']

    model.add(InputLayer(**input_parameters))

    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))

    return model


def YoloV2_MultiSize(conn, anchors, model_table='YoloV2-MultiSize', n_channels=3, width=416, height=416, scale=1.0 / 255,
                     random_mutation=None, act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                     coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                     n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                     coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                     detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                     num_to_force_coord=None, random_flip=None, random_crop=None):
    '''
    Generates a deep learning model with the Yolov2 architecture.

    The model is same as Yolov2 proposed in original paper. In addition to
    Yolov2, the model adds a passthrough layer that brings feature from an
    earlier layer to lower resolution layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information. Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)

    if input_parameters['width'] != input_parameters['height']:
        print(not_supported_feature('Non-square yolo model training', 'height=width'))
        input_parameters['height'] = input_parameters['width']

    model.add(InputLayer(**input_parameters))

    # conv1 224 416
    model.add(Conv2d(32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 112 208
    model.add(Conv2d(64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv4 56 104
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv5 56 104
    model.add(Conv2d(128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv7 28 52
    model.add(Conv2d(128, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 28 52
    model.add(Conv2d(256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv10 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv11 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv12 14 26
    model.add(Conv2d(256, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv13 14 26
    model.add(Conv2d(512, width=3, act='identity', include_bias=False, stride=1))
    pointLayer1 = BN(act=act, name='BN5_13')
    model.add(pointLayer1)
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv15 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv16 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv17 7 13
    model.add(Conv2d(512, width=1, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv18 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    # conv19 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act, name='BN6_19'))
    # conv20 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    pointLayer2 = BN(act=act, name='BN6_20')
    model.add(pointLayer2)

    # conv21 7 26 * 26 * 512 -> 26 * 26 * 64
    model.add(Conv2d(64, width=1, act='identity', include_bias=False, stride=1, src_layers=[pointLayer1]))
    model.add(BN(act=act))
    # reshape 26 * 26 * 64 -> 13 * 13 * 256
    pointLayer3 = Reshape(act='identity', width=grid_number, height=grid_number, depth=256, name='reshape1')
    model.add(pointLayer3)

    # concat
    model.add(Concat(act='identity', src_layers=[pointLayer2, pointLayer3]))

    # conv22 7 13
    model.add(Conv2d(1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(
        Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))

    return model


def Tiny_YoloV2(conn, anchors, model_table='Tiny-Yolov2', n_channels=3, width=416, height=416, scale=1.0 / 255,
                random_mutation=None, act='leaky', act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=5, do_sqrt=True, grid_number=13,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False, match_anchor_size=None,
                num_to_force_coord=None, random_flip=None, random_crop=None):
    '''
    Generate a deep learning model with the Tiny Yolov2 architecture.

    Tiny Yolov2 is a very small model of Yolov2, so that it includes fewer
    numbers of convolutional layer and batch normalization layer.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    anchors : list
        Specifies the anchor box values.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 416
    height : int, optional
        Specifies the height of the input layer.
        Default: 416
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
    act : string, optional
        Specifies the activation function for the batch normalization layers.
        Default: 'leaky'
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 5
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 13
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information.
        Default: False
    match_anchor_size : bool, optional
        Whether to force the predicted box match the anchor boxes in sizes for all predictions
    num_to_force_coord : int, optional
        The number of leading chunk of images in training when the algorithm forces predicted objects
        in each grid to be equal to the anchor box sizes, and located at the grid center
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1612.08242.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)
    if input_parameters['width'] != input_parameters['height']:
        print(not_supported_feature('Non-square yolo model training', 'height=width'))
        input_parameters['height'] = input_parameters['width']

    model.add(InputLayer(**input_parameters))

    # conv1 416 448
    model.add(Conv2d(n_filters=16, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 208 224
    model.add(Conv2d(n_filters=32, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 104 112
    model.add(Conv2d(n_filters=64, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv4 52 56
    model.add(Conv2d(n_filters=128, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv5 26 28
    model.add(Conv2d(n_filters=256, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 13 14
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    model.add(Pooling(width=2, height=2, stride=1, pool='max'))
    # conv7 13
    model.add(Conv2d(n_filters=1024, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))
    # conv8 13
    model.add(Conv2d(n_filters=512, width=3, act='identity', include_bias=False, stride=1))
    model.add(BN(act=act))

    model.add(Conv2d((n_classes + 5) * predictions_per_grid, width=1, act='identity', include_bias=False, stride=1))

    model.add(Detection(act=act_detection, detection_model_type='yolov2', anchors=anchors,
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes,
                        match_anchor_size=match_anchor_size, num_to_force_coord=num_to_force_coord))
    return model


def YoloV1(conn, model_table='YoloV1', n_channels=3, width=448, height=448, scale=1.0 / 255,
           random_mutation=None, act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
           coord_type='YOLO', max_label_per_image=30, max_boxes=30,
           n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
           coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
           detection_threshold=None, iou_threshold=None, random_boxes=False, random_flip=None, random_crop=None):
    '''
    Generates a deep learning model with the Yolo V1 architecture.

    Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information.
        Default: False
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)
    if input_parameters['width'] != input_parameters['height']:
        print(not_supported_feature('Non-square yolo model training', 'height=width'))
        input_parameters['height'] = input_parameters['width']

    model.add(InputLayer(**input_parameters))

    # conv1 448
    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv2 224
    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv3 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    # conv4 112
    model.add(Conv2d(64, width=1, act=act, include_bias=False, stride=1))
    # conv5 112
    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv6 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    # conv7 56
    model.add(Conv2d(128, width=1, act=act, include_bias=False, stride=1))
    # conv8 56
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv9 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv10 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv11 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    # conv12 28
    model.add(Conv2d(256, width=1, act=act, include_bias=False, stride=1))
    # conv13 28
    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))
    # conv14 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv15 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv16 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv17 14
    model.add(Conv2d(512, width=1, act=act, include_bias=False, stride=1))
    # conv18 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))

    # conv19 14
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv20 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=2))
    # conv21 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv22 7
    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    # conv23 7
    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))
    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(act = act_detection, detection_model_type = 'yolov1',
                        softmax_for_class_prob = softmax_for_class_prob, coord_type = coord_type,
                        class_number = n_classes, grid_number = grid_number,
                        predictions_per_grid = predictions_per_grid, do_sqrt = do_sqrt, coord_scale = coord_scale,
                        object_scale = object_scale, prediction_not_a_object_scale = prediction_not_a_object_scale,
                        class_scale = class_scale, detection_threshold = detection_threshold,
                        iou_threshold = iou_threshold, random_boxes = random_boxes,
                        max_label_per_image = max_label_per_image, max_boxes = max_boxes))

    return model


def Tiny_YoloV1(conn, model_table='Tiny-YoloV1', n_channels=3, width=448, height=448, scale=1.0 / 255,
                random_mutation=None, act='leaky', dropout=0, act_detection='AUTO', softmax_for_class_prob=True,
                coord_type='YOLO', max_label_per_image=30, max_boxes=30,
                n_classes=20, predictions_per_grid=2, do_sqrt=True, grid_number=7,
                coord_scale=None, object_scale=None, prediction_not_a_object_scale=None, class_scale=None,
                detection_threshold=None, iou_threshold=None, random_boxes=False, random_flip=None, random_crop=None):
    '''
    Generates a deep learning model with the Tiny Yolov1 architecture.

    Tiny Yolov1 is a very small model of Yolov1, so that it includes
    fewer numbers of convolutional layer.

        Parameters
    ----------
    conn : CAS
        Specifies the connection of the CAS connection.
    model_table : string, optional
        Specifies the name of CAS table to store the model.
    n_channels : int, optional
        Specifies the number of the channels (i.e., depth) of the input layer.
        Default: 3
    width : int, optional
        Specifies the width of the input layer.
        Default: 448
    height : int, optional
        Specifies the height of the input layer.
        Default: 448
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1.0 / 255
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in
        the input layer.
        Valid Values: 'none', 'random'
    act: String, optional
        Specifies the activation function to be used in the convolutional layer
        layers and the final convolution layer.
        Default: 'leaky'
    dropout: double, optional
        Specifies the drop out rate.
        Default: 0
    act_detection : string, optional
        Specifies the activation function for the detection layer.
        Valid Values: AUTO, IDENTITY, LOGISTIC, SIGMOID, TANH, RECTIFIER, RELU, SOFPLUS, ELU, LEAKY, FCMP
        Default: AUTO
    softmax_for_class_prob : bool, optional
        Specifies whether to perform Softmax on class probability per
        predicted object.
        Default: True
    coord_type : string, optional
        Specifies the format of how to represent bounding boxes. For example,
        a bounding box can be represented with the x and y locations of the
        top-left point as well as width and height of the rectangle.
        This format is the 'rect' format. We also support coco and yolo formats.
        Valid Values: 'rect', 'yolo', 'coco'
        Default: 'yolo'
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 30
    max_boxes : int, optional
        Specifies the maximum number of overall predictions allowed in the
        detection layer.
        Default: 30
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    predictions_per_grid : int, optional
        Specifies the amount of predictions will be done per grid.
        Default: 2
    do_sqrt : bool, optional
        Specifies whether to apply the SQRT function to width and height of
        the object for the cost function.
        Default: True
    grid_number : int, optional
        Specifies the amount of cells to be analyzed for an image. For example,
        if the value is 5, then the image will be divided into a 5 x 5 grid.
        Default: 7
    coord_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects exist in the grid.
    object_scale : float, optional
        Specifies the weight for object detected for the cost function in
        the detection layer.
    prediction_not_a_object_scale : float, optional
        Specifies the weight for the cost function in the detection layer,
        when objects do not exist in the grid.
    class_scale : float, optional
        Specifies the weight for the class of object detected for the cost
        function in the detection layer.
    detection_threshold : float, optional
        Specifies the threshold for object detection.
    iou_threshold : float, optional
        Specifies the IOU Threshold of maximum suppression in object detection.
    random_boxes : bool, optional
        Randomizing boxes when loading the bounding box information.
        Default: False
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

    Returns
    -------
    :class:`Sequential`

    References
    ----------
    https://arxiv.org/pdf/1506.02640.pdf

    '''

    model = Sequential(conn=conn, model_table=model_table)

    parameters = locals()
    input_parameters = get_layer_options(input_layer_options, parameters)
    if input_parameters['width'] != input_parameters['height']:
        print(not_supported_feature('Non-square yolo model training', 'height=width'))
        input_parameters['height'] = input_parameters['width']

    model.add(InputLayer(**input_parameters))

    model.add(Conv2d(16, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(32, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(64, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(128, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(512, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(1024, width=3, act=act, include_bias=False, stride=1))
    model.add(Pooling(width=2, height=2, stride=2, pool='max'))

    model.add(Conv2d(256, width=3, act=act, include_bias=False, stride=1, dropout=dropout))

    model.add(Dense(n=(n_classes + (5 * predictions_per_grid)) * grid_number * grid_number, act='identity'))

    model.add(Detection(act=act_detection, detection_model_type='yolov1',
                        softmax_for_class_prob=softmax_for_class_prob, coord_type=coord_type,
                        class_number=n_classes, grid_number=grid_number,
                        predictions_per_grid=predictions_per_grid, do_sqrt=do_sqrt, coord_scale=coord_scale,
                        object_scale=object_scale, prediction_not_a_object_scale=prediction_not_a_object_scale,
                        class_scale=class_scale, detection_threshold=detection_threshold,
                        iou_threshold=iou_threshold, random_boxes=random_boxes,
                        max_label_per_image=max_label_per_image, max_boxes=max_boxes))

    return model

