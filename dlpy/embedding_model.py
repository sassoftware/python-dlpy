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

''' The Embedding Model class adds training, evaluation,feature analysis routines for learning embedding '''

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import sys

from dlpy import Model
from dlpy.layers import Input, Dense, EmbeddingLoss
from .utils import image_blocksize, unify_keys, input_table_check, random_name, check_caslib, caslibify
from .utils import filter_by_image_id, filter_by_filename, isnotebook
from dlpy.timeseries import TimeseriesTable
from dlpy.timeseries import _get_first_obs, _get_last_obs, _combine_table, _prepare_next_input
from dlpy.utils import DLPyError, Box, DLPyDict
from dlpy.lr_scheduler import _LRScheduler, FixedLR, StepLR, FCMPLR


class EmbeddingModel(Model):
    input_layer_name_prefix = 'InputLayer_'
    embedding_layer_name_prefix = 'EmbeddingLayer_'
    embedding_loss_layer_name = 'EmbeddingLossLayer'
    number_of_branches = 0

    @classmethod
    def build_embedding_model(cls, branch, model_table=None, embedding_model_type='Siamese',
                              embedding_layer=None, margin=None):
        '''

        Build an embedding model based on a given model branch and model type

        Parameters
        ----------
        conn : CAS
            Specifies the CAS connection object.
        model_table : string or dict or CAS table, optional
            Specifies the CAS table to store the deep learning model.
            Default: None
        branch : Model
            Specifies the base model that is used as branches for embedding model.
        embedding_model_type : string, optional
            Specifies the embedding model type that the created table will be applied for training.
            Valid values: Siamese, Triplet, and Quartet.
            Default: Siamese
        embedding_layer: Layer, optional
            Specifies a dense layer as the embedding layer. For instance, Dense(n=10, act='identity') defines
            the embedding dimension is 10. When it is not given, the last layer (except the task layers)
            in the branch model will be used as the embedding layer.
        margin: double, optional
            Specifies the margin value used by the embedding model. When it is not given, for Siamese, margin is 2.0.
            Otherwise, margin is 0.0.

        Returns
        -------
        :class:`Model`

        '''

        # check the branch type
        if not isinstance(branch, Model):
            raise DLPyError('The branch option must contain a valid model')

        # the branch must be built using functional APIs
        # only functional model has the attr output_layers
        if not hasattr(branch, 'output_layers'):
            print("NOTE: Convert the branch model into a functional model.")
            branch_tensor = branch.to_functional_model()
        else:
            branch_tensor = deepcopy(branch)

        # always reset this local tensor to 0
        branch_tensor.number_of_instances = 0

        # the branch cannot contain other task layers
        if len(branch_tensor.output_layers) != 1:
            raise DLPyError('The branch model cannot contain more than one output layer')
        elif branch_tensor.output_layers[0].type == 'output':
            print("NOTE: Remove the output layers from the model.")
            branch_tensor.layers.remove(branch_tensor.output_layers[0])
            branch_tensor.output_layers[0] = branch_tensor.layers[-1]

        # check embedding_model_type
        if embedding_model_type.lower() not in ['siamese', 'triplet', 'quartet']:
            raise DLPyError('Only Siamese, Triplet, and Quartet are valid.')

        if embedding_model_type.lower() == 'siamese':
            if margin is None:
                margin = 2.0
            cls.number_of_branches = 2
        elif embedding_model_type.lower() == 'triplet':
            if margin is None:
                margin = 0.0
            cls.number_of_branches = 3
        elif embedding_model_type.lower() == 'quartet':
            if margin is None:
                margin = 0.0
            cls.number_of_branches = 4

        # build the branches
        input_layers = []
        branch_layers = []
        for i_branch in range(cls.number_of_branches):
            temp_input_layer = Input(**branch_tensor.layers[0].config, name=cls.input_layer_name_prefix + str(i_branch))
            temp_branch = branch_tensor(temp_input_layer)  # return a list of tensors
            if embedding_layer:
                temp_embed_layer = deepcopy(embedding_layer)
                temp_embed_layer.name = cls.embedding_layer_name_prefix + str(i_branch)
                temp_branch = temp_embed_layer(temp_branch)
                # change tensor to a list
                temp_branch = [temp_branch]
            else:
                # change the last layer name to the embedding layer name
                temp_branch[-1]._op.name = cls.embedding_layer_name_prefix + str(i_branch)

            # append these layers to the current branch
            input_layers.append(temp_input_layer)
            branch_layers = branch_layers + temp_branch

        # add the embedding loss layer
        loss_layer = EmbeddingLoss(margin=margin, name=cls.embedding_loss_layer_name)(branch_layers)

        # create the model DAG using all the above model information
        model = EmbeddingModel(branch.conn, model_table=model_table, inputs=input_layers, outputs=loss_layer)

        # sharing weights
        # get all layer names from one branch
        num_l = int((len(model.layers) - 1) / cls.number_of_branches)
        br1_name = [i.name for i in model.layers[:num_l - 1]]

        # build the list that contain the shared layers
        share_list = []
        n_id = 0
        n_to = n_id + cls.number_of_branches
        for l in br1_name[1:]:
            share_list.append({l: [l + '_' + str(i + 1) for i in range(n_id + 1, n_to)]})

        # add embedding layers
        share_list.append({cls.embedding_layer_name_prefix + str(0):
                               [cls.embedding_layer_name_prefix + str(i)
                                for i in range(1, cls.number_of_branches)]})

        model.share_weights(share_list)

        model.compile()

        return model

    def fit_embedding_model(self):
        # super(self).fit()

        pass

    def deploy_embedding_model(self):
        pass
