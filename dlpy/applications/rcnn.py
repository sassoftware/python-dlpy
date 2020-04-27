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


from dlpy.model import Model
from dlpy.sequential import Sequential
from dlpy.layers import (Conv2d, Input, Pooling, RegionProposal, ROIPooling, Dense, FastRCNN)
from .application_utils import get_layer_options, input_layer_options, rpn_layer_options, fast_rcnn_options
from dlpy.utils import DLPyError


def Faster_RCNN(conn, model_table='Faster_RCNN', n_channels=3, width=1000, height=496, scale=1,
                norm_stds=None, offsets=(102.9801, 115.9465, 122.7717), random_mutation=None,
                n_classes=20, anchor_num_to_sample=256, anchor_ratio=[0.5, 1, 2], anchor_scale=[8, 16, 32],
                base_anchor_size=16, coord_type='coco', max_label_per_image=200, proposed_roi_num_train=2000,
                proposed_roi_num_score=300, roi_train_sample_num=128, roi_pooling_height=7, roi_pooling_width=7,
                nms_iou_threshold=0.3, detection_threshold=0.5, max_object_num=50, number_of_neurons_in_fc=4096,
                backbone='vgg16', random_flip=None, random_crop=None):
    '''
    Generates a deep learning model with the faster RCNN architecture.

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
        Default: 1000
    height : int, optional
        Specifies the height of the input layer.
        Default: 496
    scale : double, optional
        Specifies a scaling factor to be applied to each pixel intensity values.
        Default: 1
    norm_stds : double or iter-of-doubles, optional
        Specifies a standard deviation for each channel in the input data.
        The final input data is normalized with specified means and standard deviations.
    offsets : double or iter-of-doubles, optional
        Specifies an offset for each channel in the input data. The final input
        data is set after applying scaling and subtracting the specified offsets.
    random_mutation : string, optional
        Specifies how to apply data augmentations/mutations to the data in the
        input layer.
        Valid Values: 'none', 'random'
    n_classes : int, optional
        Specifies the number of classes. If None is assigned, the model will
        automatically detect the number of classes based on the training set.
        Default: 20
    anchor_num_to_sample : int, optional
        Specifies the number of anchors to sample for training the region proposal network
        Default: 256
    anchor_ratio : iter-of-float
        Specifies the anchor height and width ratios (h/w) used.
    anchor_scale : iter-of-float
        Specifies the anchor scales used based on base_anchor_size
    base_anchor_size : int, optional
        Specifies the basic anchor size in width and height (in pixels) in the original input image dimension
        Default: 16
    coord_type : int, optional
        Specifies the coordinates format type in the input label and detection result.
        Valid Values: RECT, COCO, YOLO
        Default: COCO
    proposed_roi_num_score: int, optional
        Specifies the number of ROI (Region of Interest) to propose in the scoring phase
        Default: 300
    proposed_roi_num_train: int, optional
        Specifies the number of ROI (Region of Interest) to propose used for RPN training, and also the pool to
        sample from for FastRCNN Training in the training phase
        Default: 2000
    roi_train_sample_num: int, optional
        Specifies the number of ROIs(Regions of Interests) to sample after NMS(Non-maximum Suppression)
        is performed in the training phase.
        Default: 128
    roi_pooling_height : int, optional
        Specifies the output height of the region pooling layer.
        Default: 7
    roi_pooling_width : int, optional
        Specifies the output width of the region pooling layer.
        Default: 7
    max_label_per_image : int, optional
        Specifies the maximum number of labels per image in the training.
        Default: 200
    nms_iou_threshold: float, optional
        Specifies the IOU threshold of maximum suppression in object detection
        Default: 0.3
    detection_threshold : float, optional
        Specifies the threshold for object detection.
        Default: 0.5
    max_object_num: int, optional
        Specifies the maximum number of object to detect
        Default: 50
    number_of_neurons_in_fc: int, or list of int, optional
        Specifies the number of neurons in the last two fully connected layers. If one int is set, then
        both of the layers will have the same values. If a list is set, then the layers get different
        number of neurons.
        Default: 4096
    backbone: string, optional
        Specifies the architecture to be used as the feature extractor.
        Valid values: vgg16
        Default: vgg16, resnet50, resnet18, resnet34, mobilenetv1, mobilenetv2
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
    https://arxiv.org/abs/1506.01497

    '''
    # calculate number of anchors that equal to product of length of anchor_ratio and length of anchor_scale
    num_anchors = len(anchor_ratio) * len(anchor_scale)
    parameters = locals()
    # get parameters of input, rpn, fast_rcnn layer
    input_parameters = get_layer_options(input_layer_options, parameters)
    rpn_parameters = get_layer_options(rpn_layer_options, parameters)
    fast_rcnn_parameters = get_layer_options(fast_rcnn_options, parameters)
    inp = Input(**input_parameters, name='data')

    if backbone.lower() == 'vgg16':
        # backbone is VGG16 model
        conv1_1 = Conv2d(n_filters=64, width=3, height=3, stride=1, name='conv1_1')(inp)
        conv1_2 = Conv2d(n_filters=64, width=3, height=3, stride=1, name='conv1_2')(conv1_1)
        pool1 = Pooling(width=2, height=2, stride=2, pool='max', name='pool1')(conv1_2)

        conv2_1 = Conv2d(n_filters=128, width=3, height=3, stride=1, name='conv2_1')(pool1)
        conv2_2 = Conv2d(n_filters=128, width=3, height=3, stride=1, name='conv2_2')(conv2_1)
        pool2 = Pooling(width=2, height=2, stride=2, pool='max')(conv2_2)

        conv3_1 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_1')(pool2)
        conv3_2 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_2')(conv3_1)
        conv3_3 = Conv2d(n_filters=256, width=3, height=3, stride=1, name='conv3_3')(conv3_2)
        pool3 = Pooling(width=2, height=2, stride=2, pool='max')(conv3_3)

        conv4_1 = Conv2d(n_filters=512, width=3, height=3, stride = 1, name = 'conv4_1')(pool3)
        conv4_2 = Conv2d(n_filters=512, width=3, height=3, stride = 1, name = 'conv4_2')(conv4_1)
        conv4_3 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv4_3')(conv4_2)
        pool4 = Pooling(width=2, height=2, stride=2, pool='max')(conv4_3)

        conv5_1 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_1')(pool4)
        conv5_2 = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_2')(conv5_1)
        # feature of Conv5_3 is used to generate region proposals
        last_layer_in_backbone = Conv2d(n_filters=512, width=3, height=3, stride=1, name='conv5_3')(conv5_2)
        # two convolutions build on top of conv5_3 and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)

        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)

        # given ROIs, crop on conv5_3 and resize the feature to the same size
        roipool1 = ROIPooling(output_height=roi_pooling_height,
                              output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone.shape[0]/width,
                              name='roi_pooling')([last_layer_in_backbone, rp1])

    elif backbone.lower() == 'resnet50':

        from .resnet import ResNet50_SAS
        backbone = ResNet50_SAS(conn, width=width, height=height)
        backbone.layers[-2].src_layers
        backbone_with_last = backbone.to_functional_model(stop_layers=backbone.layers[-2])
        last_layer_in_backbone = backbone_with_last(inp)
        # two convolutions build on top of f_ex and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
        roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone[0].shape[0]/height,
                              name='roi_pooling')([last_layer_in_backbone[0], rp1])

    elif backbone.lower() == 'resnet34':
        from .resnet import ResNet34_SAS
        backbone = ResNet34_SAS(conn, width=width, height=height)
        backbone.layers[-2].src_layers
        backbone_with_last = backbone.to_functional_model(stop_layers=backbone.layers[-2])
        last_layer_in_backbone = backbone_with_last(inp)
        # two convolutions build on top of f_ex and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
        roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone[0].shape[0]/height,
                              name='roi_pooling')([last_layer_in_backbone[0], rp1])

    elif backbone.lower() == 'resnet18':
        from .resnet import ResNet18_SAS
        backbone = ResNet18_SAS(conn, width=width, height=height)
        backbone.layers[-2].src_layers
        backbone_with_last = backbone.to_functional_model(stop_layers=backbone.layers[-2])
        last_layer_in_backbone = backbone_with_last(inp)
        # two convolutions build on top of f_ex and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
        roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone[0].shape[0]/height,
                              name='roi_pooling')([last_layer_in_backbone[0], rp1])

    elif backbone.lower() == 'mobilenetv1':
        from .mobilenet import MobileNetV1
        backbone = MobileNetV1(conn, width=width, height=height)
        backbone.layers[-2].src_layers
        backbone_with_last = backbone.to_functional_model(stop_layers=backbone.layers[-2])
        last_layer_in_backbone = backbone_with_last(inp)
        # two convolutions build on top of f_ex and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
        roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone[0].shape[0]/height,
                              name='roi_pooling')([last_layer_in_backbone[0], rp1])

    elif backbone.lower() == 'mobilenetv2':
        from .mobilenet import MobileNetV2
        backbone = MobileNetV2(conn, width=width, height=height)
        backbone.layers[-2].src_layers
        backbone_with_last = backbone.to_functional_model(stop_layers=backbone.layers[-2])
        last_layer_in_backbone = backbone_with_last(inp)
        # two convolutions build on top of f_ex and reduce feature map depth to 6*number_anchors
        rpn_conv = Conv2d(width=3, n_filters=512, name='rpn_conv_3x3')(last_layer_in_backbone)
        rpn_score = Conv2d(act='identity', width=1, n_filters=((1 + 1 + 4) * num_anchors), name='rpn_score')(rpn_conv)
        # propose anchors, NMS, select anchors to train RPN, produce ROIs
        rp1 = RegionProposal(**rpn_parameters, name='rois')(rpn_score)
        roipool1 = ROIPooling(output_height=roi_pooling_height, output_width=roi_pooling_width,
                              spatial_scale=last_layer_in_backbone[0].shape[0]/height,
                              name='roi_pooling')([last_layer_in_backbone[0], rp1])
    else:
        raise DLPyError('We are not supporting this backbone yet.')

    # fully connect layer to extract the feature of ROIs
    if number_of_neurons_in_fc is None:
        fc6 = Dense(n=4096, act='relu', name='fc6')(roipool1)
        fc7 = Dense(n=4096, act='relu', name='fc7')(fc6)
    else:
        if isinstance(number_of_neurons_in_fc, list):
            if len(number_of_neurons_in_fc) > 1:
                fc6 = Dense(n=number_of_neurons_in_fc[0], act='relu', name='fc6')(roipool1)
                fc7 = Dense(n=number_of_neurons_in_fc[1], act='relu', name='fc7')(fc6)
            else:
                fc6 = Dense(n=number_of_neurons_in_fc[0], act='relu', name='fc6')(roipool1)
                fc7 = Dense(n=number_of_neurons_in_fc[0], act='relu', name='fc7')(fc6)
        else:
            fc6 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc6')(roipool1)
            fc7 = Dense(n=number_of_neurons_in_fc, act='relu', name='fc7')(fc6)
    # classification tensor
    cls1 = Dense(n=n_classes+1, act='identity', name='cls_score')(fc7)
    # regression tensor(second stage bounding box regression)
    reg1 = Dense(n=(n_classes+1)*4, act='identity', name='bbox_pred')(fc7)
    # task layer receive cls1, reg1 and rp1(ground truth). Train the second stage.
    fr1 = FastRCNN(**fast_rcnn_parameters, class_number=n_classes, name='fastrcnn')([cls1, reg1, rp1])
    faster_rcnn = Model(conn, inp, fr1, model_table=model_table)
    faster_rcnn.compile()
    return faster_rcnn
