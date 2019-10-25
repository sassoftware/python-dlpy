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

import six

# input layer option will be found in model function's local parameters
input_layer_options = ['n_channels', 'width', 'height', 'nominals', 'std', 'scale', 'offsets',
                       'dropout', 'random_flip', 'random_crop', 'random_mutation', 'norm_stds']

# RPN layer option will be found in model function's local parameters
rpn_layer_options = ['anchor_ratio', 'anchor_scale', 'anchor_num_to_sample', 'base_anchor_size',
                     'coord_type', 'do_RPN_only', 'max_label_per_image', 'proposed_roi_num_score',
                     'proposed_roi_num_train', 'roi_train_sample_num']
# Fast RCNN option will be found in model function's local parameters
fast_rcnn_options = ['detection_threshold', 'max_label_per_image', 'max_object_num', 'nms_iou_threshold']


def get_layer_options(layer_options, local_options):
    """
    Get parameters belonging to a certain type of layer.

    Parameters
    ----------
    layer_options : list of String
        Specifies parameters of the layer.
    local_options : list of dictionary
        Specifies local parameters in a model function.

    """
    layer_options_dict = {}
    for key, value in six.iteritems(local_options):
        if key in layer_options:
            layer_options_dict[key] = value
    return layer_options_dict


def not_supported_feature(feature, action=None):

    '''
    Returns a templated note for the specified feature indicating that it is not supported.

    Parameters
    ----------

    feature : string
        Specifies the feature that is not supported. This will be inserted into the templated note message.

    action : string, optional
        Specifies if an action is taken for the not supported feature.

    Returns
    -------
    warning_message : str

    '''

    if action is None:
        return '*** NOTE: '+feature+' is not supported yet.'
    else:
        return '*** NOTE: '+feature+' is not supported yet. We took the following action: '+action

