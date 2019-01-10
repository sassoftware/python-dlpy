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

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.

import os
import swat
import swat.utils.testing as tm
import numpy as np
import numpy.random as nr
import pandas as pd
import unittest
from dlpy.metrics import *
from dlpy.utils import random_name, get_server_path_sep
from sklearn.metrics import roc_auc_score as skroc
from sklearn.metrics import f1_score as skf1
from sklearn.metrics import average_precision_score as skaps
from sklearn.metrics import confusion_matrix as skcm
from sklearn.metrics import accuracy_score as skas
from sklearn.metrics import explained_variance_score as skevs
from sklearn.metrics import mean_absolute_error as skmae
from sklearn.metrics import mean_squared_error as skmse
from sklearn.metrics import r2_score as skr2sc
from sklearn.metrics import mean_squared_log_error as skmsle

def _random_weighted_select(prob_matrix, item=None, axis=1):
    prob_cumsum = prob_matrix.cumsum(axis=axis)
    uniform_random = nr.rand(prob_matrix.shape[1-axis])
    uniform_random = np.expand_dims(uniform_random, axis=axis)
    select_idx = (prob_cumsum < uniform_random).sum(axis=axis)
    if item is not None:
        if not isinstance(item, np.ndarray):
            item = np.array(item)  
        return item[select_idx]
    else:
        return select_idx
    
def _create_classification_table(nclass, nrow, alpha=None, seed=1234,
                                 true_label='target', pred_label='p_target'):
    
    if alpha is None:
        alpha = [1]*nclass
    
    nr.seed(seed)
    prob_matrix = nr.dirichlet(alpha, size=nrow)
    target = _random_weighted_select(prob_matrix).reshape(-1, 1)
    p_target = prob_matrix.argmax(axis=1).reshape(-1, 1)
    classification_matrix = np.hstack((prob_matrix, target, p_target))
    colnames = ['p_' + str(i) for i in range(nclass)] + [true_label, pred_label]
    
    return pd.DataFrame(classification_matrix, columns=colnames)

def _create_regression_table(nrow, seed=1234, 
                             true_label='target', pred_label='p_target'):
    
    nr.seed(seed)
    mean_value = nr.normal(loc=10, scale=2, size=(nrow, 1))
    error = nr.normal(loc=0, scale=1, size=(nrow, 1))
    true_value = mean_value + error
    regression_matrix = np.hstack((true_value, mean_value))
    regression_matrix = np.abs(regression_matrix)

    return pd.DataFrame(regression_matrix, columns=[true_label, pred_label])


class TestMetrics(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    conn = None
    server_sep = '/'
    data_dir = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.conn = swat.CAS()
        self.server_sep = get_server_path_sep(self.conn)

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep
        
        pandas_class_table1 = _create_classification_table(5, 500)
        pandas_class_table2 = _create_classification_table(2, 500)
        pandas_regression_table1 = _create_regression_table(500)
        
        self.class_table1 = self.conn.upload_frame(pandas_class_table1, 
                                                   casout=dict(name=random_name(name='class1_'),
                                                               replace=True))
        self.class_table2 = self.conn.upload_frame(pandas_class_table2, 
                                                   casout=dict(name=random_name(name='class2_'),
                                                               replace=True))
        self.reg_table1 = self.conn.upload_frame(pandas_regression_table1, 
                                                   casout=dict(name=random_name(name='reg1_'),
                                                               replace=True))

    def tearDown(self):
        # tear down tests
        try:
            self.conn.endsession()
        except swat.SWATError:
            pass
        del self.conn
        swat.reset_option()
        
    def test_accuracy_score(self):
        
        local_class1 = self.class_table1.to_frame()
        skas_score1  = skas(local_class1.target, local_class1.p_target, normalize=False)
        skas_score1_norm = skas(local_class1.target, local_class1.p_target, normalize=True)
        
        dlpyas_score1 = accuracy_score(self.class_table1, 'target', 'p_target', normalize=False)
        dlpyas_score1_norm = accuracy_score(self.class_table1, 'target', 'p_target', normalize=True)
        
        self.assertEqual(skas_score1, dlpyas_score1)
        self.assertEqual(skas_score1_norm, dlpyas_score1_norm)
        
        local_class2 = self.class_table2.to_frame()
        skas_score2  = skas(local_class2.target, local_class2.p_target, normalize=False)
        skas_score2_norm = skas(local_class2.target, local_class2.p_target, normalize=True)
        
        dlpyas_score2 = accuracy_score(self.class_table2, 'target', 'p_target', normalize=False)
        dlpyas_score2_norm = accuracy_score(self.class_table2, 'target', 'p_target', normalize=True)
        
        self.assertEqual(skas_score2, dlpyas_score2)
        self.assertEqual(skas_score2_norm, dlpyas_score2_norm)
        
    def test_confusion_matrix(self):
        
        local_class1 = self.class_table1.to_frame()
        skcm_matrix1  = skcm(local_class1.target, local_class1.p_target)
        skcm_matrix2  = skcm(local_class1.target, local_class1.p_target, labels=[1,3,4])
        
        dlpycm_matrix1 = confusion_matrix(self.class_table1, 'target', 'p_target')
        dlpycm_matrix2 = confusion_matrix(self.class_table1, 'target', 'p_target', labels=[1,3,4])
        
        self.assertTrue(np.array_equal(skcm_matrix1, dlpycm_matrix1.values))
        self.assertTrue(np.array_equal(skcm_matrix2, dlpycm_matrix2.values))
   
        
    def test_plot_roc(self):        
        ax1 = plot_roc(self.class_table2, 'target', 'p_1', pos_label=1)
        ax2 = plot_roc(self.class_table2, 'target', 'p_0', pos_label=0)
        ax3 = plot_roc(self.class_table2, 'target', 'p_0', pos_label=0,
                       fontsize_spec={'xlabel':20})
        ax4 = plot_roc(self.class_table1, 'target', 'p_3', pos_label=3)
        
    def test_plot_precision_recall(self):        
        ax1 = plot_precision_recall(self.class_table2, 'target', 'p_1', pos_label=1)
        ax2 = plot_precision_recall(self.class_table2, 'target', 'p_0', pos_label=0)
        ax3 = plot_precision_recall(self.class_table2, 'target', 'p_0', pos_label=0,
                               fontsize_spec={'xlabel':20})
        ax4 = plot_precision_recall(self.class_table1, 'target', 'p_3', pos_label=3)
        
    def test_roc_auc_score(self):
        
        local_class2 = self.class_table2.to_frame()
        skauc_score = skroc(local_class2.target, local_class2.p_1)       
        dlpyauc_score = roc_auc_score(self.class_table2, 'target', 'p_1', pos_label=1)        
        self.assertAlmostEqual(skauc_score, dlpyauc_score, places=4)
        
    def test_average_precision_score(self):
        
        local_class2 = self.class_table2.to_frame()
        skaps_score = skaps(local_class2.target, local_class2.p_1, pos_label=1)       
        dlpyaps_score = average_precision_score(self.class_table2, 'target', 'p_1', pos_label=1) 
        dlpyaps_score2 =  average_precision_score(self.class_table2, 'target', 'p_1', pos_label=1, 
                                                  interpolate=True)
        self.assertAlmostEqual(skaps_score, dlpyaps_score, places=4)
        
    def test_f1_score(self):

        local_class2 = self.class_table2.to_frame()
        skf1_score1 = skf1(local_class2.target, local_class2.p_target, pos_label=1)    
        dlpyf1_score1 = f1_score(self.class_table2, 'target', 'p_target', pos_label=1)
        
        self.assertAlmostEqual(skf1_score1, dlpyf1_score1)
        
    def test_explained_variance_score(self):
        
        local_reg1 = self.reg_table1.to_frame()
        skevs_score1 = skevs(local_reg1.target, local_reg1.p_target)    
        dlpyevs_score1 = explained_variance_score(self.reg_table1, 'target', 'p_target')
        
        self.assertAlmostEqual(skevs_score1, dlpyevs_score1)        
        
    def test_mean_absolute_error(self):
        
        local_reg1 = self.reg_table1.to_frame()
        skmae_score1 = skmae(local_reg1.target, local_reg1.p_target)    
        dlpymae_score1 = mean_absolute_error(self.reg_table1, 'target', 'p_target')
        
        self.assertAlmostEqual(skmae_score1, dlpymae_score1)
        
    def test_mean_squared_error(self):
        
        local_reg1 = self.reg_table1.to_frame()
        skmse_score1 = skmse(local_reg1.target, local_reg1.p_target)    
        dlpymse_score1 = mean_squared_error(self.reg_table1, 'target', 'p_target')
        
        self.assertAlmostEqual(skmse_score1, dlpymse_score1)
        
    def test_mean_squared_log_error(self):
        
        local_reg1 = self.reg_table1.to_frame()
        skmsle_score1 = skmsle(local_reg1.target, local_reg1.p_target)    
        dlpymsle_score1 = mean_squared_log_error(self.reg_table1, 'target', 'p_target')
        
        self.assertAlmostEqual(skmsle_score1, dlpymsle_score1)
        
    def test_r2_score(self):
        
        local_reg1 = self.reg_table1.to_frame()
        skr2sc_score1 = skr2sc(local_reg1.target, local_reg1.p_target)    
        dlpyr2sc_score1 = r2_score(self.reg_table1, 'target', 'p_target')
        
        self.assertAlmostEqual(skr2sc_score1, dlpyr2sc_score1)    
        
        
        
        
        