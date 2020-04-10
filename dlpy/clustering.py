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

''' The Cluster class adds building, training and evaluation for deep clustering '''

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections
import sys
from .utils import image_blocksize, unify_keys, input_table_check, random_name, check_caslib, caslibify
from .utils import filter_by_image_id, filter_by_filename, isnotebook
from dlpy.utils import DLPyError, Box, DLPyDict
from dlpy.lr_scheduler import _LRScheduler, FixedLR, StepLR, FCMPLR
from dlpy.network import Network
from sklearn.utils import linear_assignment

class Clustering(Model):
    def cluster_acc(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_assignment(w.max() - w)
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    def build_clustering_model(cls, model, src_layer, max_clusters, alpha=1.0):
        '''
        Build an deep clustering model based on a given feature extraction model

        parameters
        ----------
        conn : CAS
            Specifies the CAS connection object.
        model : Model
            Specifies the feature extraction model.
        src_layer : Layer
            Specifies the source layer of the clustering layer.
        max_clusters : Integer
            Specifies the maximum number of clusters for the cluster layer.
        alpha : double, optional
            Specifies the degree of freedom of the t-distribution kernel for the cluster layer.

        Returns
        -------
        :class:`Model`

        '''

        # check the model type
        if not isinstance(branch, Model):
            raise DLPyError('The branch option must contain a valid model')

