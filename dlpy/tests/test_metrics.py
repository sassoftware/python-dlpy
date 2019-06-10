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
import numpy.random as nr
import numpy as np
import pandas as pd
import itertools
import unittest
import dlpy
from dlpy.metrics import *
from dlpy.utils import random_name, get_server_path_sep


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
    

def _create_id_matrix(nrow, ncol, seed=1234):
    
    uniq_per_col = np.ceil(np.power(nrow, 1/ncol)).astype(np.int64)
    
    id_matrix = np.array([x for x in itertools.product(range(uniq_per_col), repeat=ncol)])
    
    np.random.seed(seed=seed)
    if id_matrix.shape[0]>nrow:
        id_matrix = id_matrix[np.random.choice(id_matrix.shape[0], nrow, replace=False), :]
        
    return id_matrix


def _create_classification_table(nclass, nrow, id_vars=None, alpha=None, seed=1234,
                                 true_label='target', pred_label='p_target'):
    
    if alpha is None:
        alpha = [1]*nclass
    
    nr.seed(seed)
    prob_matrix = nr.dirichlet(alpha, size=nrow)
    target = _random_weighted_select(prob_matrix).reshape(-1, 1)
    p_target = prob_matrix.argmax(axis=1).reshape(-1, 1)
    classification_matrix = np.hstack((prob_matrix, target, p_target))
    colnames = ['p_' + str(i) for i in range(nclass)] + [true_label, pred_label]
    
    if id_vars is not None:
        if not isinstance(id_vars, list):
            id_vars = [id_vars]
        ncol = len(id_vars)
        id_matrix = _create_id_matrix(nrow, ncol)
        classification_matrix = np.hstack((classification_matrix, id_matrix))
        colnames = colnames + id_vars
        
    
    return pd.DataFrame(classification_matrix, columns=colnames)


def _create_regression_table(nrow, id_vars=None, seed=1234, true_label='target', pred_label='p_target'):
    
    nr.seed(seed)
    mean_value = nr.normal(loc=10, scale=2, size=(nrow, 1))
    error = nr.normal(loc=0, scale=1, size=(nrow, 1))
    true_value = mean_value + error
    regression_matrix = np.hstack((true_value, mean_value))
    regression_matrix = np.abs(regression_matrix)
    colnames = [true_label, pred_label]
    
    if id_vars is not None:
        if not isinstance(id_vars, list):
            id_vars = [id_vars]
        ncol = len(id_vars)
        id_matrix = _create_id_matrix(nrow, ncol)
        regression_matrix = np.hstack((regression_matrix, id_matrix))
        colnames = colnames + id_vars

    return pd.DataFrame(regression_matrix, columns=colnames)


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
        
        self.local_class1 = _create_classification_table(5, 500, seed=1234, id_vars=['id1', 'id2'])
        self.local_class2 = _create_classification_table(5, 500, seed=34, id_vars=['id1', 'id2'])
        self.local_class3 = _create_classification_table(2, 500, seed=1234, id_vars=['id1', 'id2'])
        self.local_class4 = _create_classification_table(2, 500, seed=34, id_vars=['id1', 'id2'])
        self.local_class1.sort_values(by=['id1', 'id2'], inplace=True)
        self.local_class2.sort_values(by=['id1', 'id2'], inplace=True)
        self.local_class3.sort_values(by=['id1', 'id2'], inplace=True)
        self.local_class4.sort_values(by=['id1', 'id2'], inplace=True)
        
        self.local_reg1 = _create_regression_table(500, seed=12, id_vars='id1')
        self.local_reg2 = _create_regression_table(500, seed=34, id_vars='id1')
        self.local_reg1.sort_values(by='id1', inplace=True)
        self.local_reg2.sort_values(by='id1', inplace=True)
        
        self.class_table1 = self.conn.upload_frame(self.local_class1, 
                                                   casout=dict(name=random_name(name='class1_'), replace=True))
        self.class_table2 = self.conn.upload_frame(self.local_class2, 
                                                   casout=dict(name=random_name(name='class2_'), replace=True))
        self.class_table3 = self.conn.upload_frame(self.local_class3, 
                                                   casout=dict(name=random_name(name='class3_'), replace=True))
        self.class_table4 = self.conn.upload_frame(self.local_class4, 
                                                   casout=dict(name=random_name(name='class4_'), replace=True))
        self.reg_table1 = self.conn.upload_frame(self.local_reg1,
                                                 casout=dict(name=random_name(name='reg1_'), replace=True))
        self.reg_table2 = self.conn.upload_frame(self.local_reg2,
                                                 casout=dict(name=random_name(name='reg2_'), replace=True))

    def tearDown(self):
        # tear down tests
        try:
            self.conn.endsession()
        except swat.SWATError:
            pass
        del self.conn
        swat.reset_option()
        
    def test_accuracy_score(self):

        try:
            from sklearn.metrics import accuracy_score as skas
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skas_score1 = skas(self.local_class1.target, self.local_class1.p_target, normalize=False)
        skas_score1_norm = skas(self.local_class1.target, self.local_class1.p_target, normalize=True)
        
        dlpyas_score1 = accuracy_score('target', 'p_target', self.class_table1, normalize=False)
        dlpyas_score1_norm = accuracy_score('target', 'p_target', self.class_table1, normalize=True)
        
        self.assertEqual(skas_score1, dlpyas_score1)
        self.assertEqual(skas_score1_norm, dlpyas_score1_norm)
        
        skas_score2 = skas(self.local_class2.target, self.local_class1.p_target, normalize=False)
        skas_score2_norm = skas(self.local_class2.target, self.local_class1.p_target, normalize=True)
        
        dlpyas_score2 = accuracy_score(self.class_table2.target, self.class_table1.p_target, 
                                       normalize=False, id_vars=['id1', 'id2'])
        dlpyas_score2_norm = accuracy_score(self.class_table2.target, self.class_table1.p_target,
                                            normalize=True, id_vars=['id1', 'id2'])
        
        self.assertEqual(skas_score2, dlpyas_score2)
        self.assertEqual(skas_score2_norm, dlpyas_score2_norm)
        
        skas_score3 = skas(self.local_class3.target, self.local_class3.p_target, normalize=False)
        skas_score3_norm = skas(self.local_class3.target, self.local_class3.p_target, normalize=True)
        
        dlpyas_score3 = accuracy_score(self.class_table3.target, self.class_table3.p_target, normalize=False)
        dlpyas_score3_norm = accuracy_score(self.class_table3.target, self.class_table3.p_target, normalize=True)
        
        self.assertEqual(skas_score3, dlpyas_score3)
        self.assertEqual(skas_score3_norm, dlpyas_score3_norm)
        
    def test_confusion_matrix(self):

        try:
            from sklearn.metrics import confusion_matrix as skcm
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skcm_matrix1 = skcm(self.local_class1.target, self.local_class1.p_target)
        skcm_matrix2 = skcm(self.local_class1.target, self.local_class1.p_target, labels=[1, 3, 4])
        
        dlpycm_matrix1 = confusion_matrix(self.class_table1.target, self.class_table1.p_target)
        dlpycm_matrix2 = confusion_matrix(self.class_table1.target, self.class_table1.p_target, labels=[1, 3, 4])
        
        self.assertTrue(np.array_equal(skcm_matrix1, dlpycm_matrix1.values))
        self.assertTrue(np.array_equal(skcm_matrix2, dlpycm_matrix2.values))
        
        skcm_matrix3 = skcm(self.local_class1.target, self.local_class2.p_target)
        skcm_matrix4 = skcm(self.local_class1.target, self.local_class2.p_target, labels=[1, 3, 4])
        
        dlpycm_matrix3 = confusion_matrix(self.class_table1.target, self.class_table2.p_target,
                                          id_vars=['id1', 'id2'])
        dlpycm_matrix4 = confusion_matrix(self.class_table1.target, self.class_table2.p_target, 
                                          labels=[1, 3, 4], id_vars=['id1', 'id2'])
        
        self.assertTrue(np.array_equal(skcm_matrix3, dlpycm_matrix3.values))
        self.assertTrue(np.array_equal(skcm_matrix4, dlpycm_matrix4.values))
        
        dlpycm_matrix5 = confusion_matrix('target', 'p_target', castable=self.class_table1)
        dlpycm_matrix6 = confusion_matrix('target', 'p_target', castable=self.class_table1, labels=[1, 3, 4])
        
        self.assertTrue(np.array_equal(skcm_matrix1, dlpycm_matrix5.values))
        self.assertTrue(np.array_equal(skcm_matrix2, dlpycm_matrix6.values))

    def test_plot_roc(self):        
        ax1 = plot_roc('target', 'p_1', pos_label=1, castable=self.class_table3)
        ax2 = plot_roc('target', 'p_0', pos_label=0, castable=self.class_table3)
        ax3 = plot_roc('target', 'p_0', pos_label=0, castable=self.class_table3, 
                       fontsize_spec={'xlabel':20})
        ax4 = plot_roc(self.class_table1.target, self.class_table1.p_3, pos_label=3)
        ax5 = plot_roc(self.class_table1.target, self.class_table2.p_3, 
                       pos_label=3, id_vars=['id1', 'id2'])
        
    def test_plot_precision_recall(self):        
        ax1 = plot_precision_recall('target', 'p_1', pos_label=1, castable=self.class_table3)
        ax2 = plot_precision_recall('target', 'p_0', pos_label=0, castable=self.class_table3)
        ax3 = plot_precision_recall(self.class_table3.target, self.class_table4.p_0, pos_label=0, 
                                    fontsize_spec={'xlabel':20}, id_vars=['id1', 'id2'])
        ax4 = plot_precision_recall('target', 'p_3', pos_label=3, castable=self.class_table1)
        
    def test_roc_auc_score(self):

        try:
            from sklearn.metrics import roc_auc_score as skroc
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skauc_score1 = skroc(self.local_class3.target, self.local_class3.p_1)       
        dlpyauc_score1 = roc_auc_score(self.class_table3.target, self.class_table3.p_1, pos_label=1)        
        self.assertAlmostEqual(skauc_score1, dlpyauc_score1, places=4)
        
        skauc_score2 = skroc(self.local_class3.target, self.local_class4.p_1)       
        dlpyauc_score2 = roc_auc_score(self.class_table3.target, self.class_table4.p_1, pos_label=1, 
                                       id_vars=['id1', 'id2'])        
        self.assertAlmostEqual(skauc_score2, dlpyauc_score2, places=4)
        
    def test_average_precision_score(self):

        try:
            from sklearn.metrics import average_precision_score as skaps
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        try:
            from distutils.version import StrictVersion
        except:
            unittest.TestCase.skipTest(self, "StrictVersion issue")

        import sklearn
        if StrictVersion(sklearn.__version__) < StrictVersion('0.20.3'):
            unittest.TestCase.skipTest(self, "There is an API change in sklearn, "
                                             "this test is skipped with old versions of sklearn")

        skaps_score1 = skaps(self.local_class3.target, self.local_class3.p_1, pos_label=1)       
        dlpyaps_score1 = average_precision_score('target', 'p_1', pos_label=1, castable=self.class_table3, 
                                                 cutstep=0.0001) 
        dlpyaps_score1_inter = average_precision_score(self.class_table3.target,self.class_table3.p_1,
                                                       pos_label=1, interpolate=True)
        self.assertAlmostEqual(skaps_score1, dlpyaps_score1, places=4)
        
        skaps_score2 = skaps(self.local_class3.target, self.local_class4.p_1, pos_label=1)       
        dlpyaps_score2 = average_precision_score(self.class_table3.target,self.class_table4.p_1, pos_label=1, 
                                                 cutstep=0.0001, id_vars=['id1', 'id2']) 
        
        self.assertAlmostEqual(skaps_score2, dlpyaps_score2, places=4)        
        
    def test_f1_score(self):

        try:
            from sklearn.metrics import f1_score as skf1
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skf1_score1 = skf1(self.local_class3.target, self.local_class3.p_target, pos_label=1)    
        dlpyf1_score1 = f1_score(self.class_table3.target, self.class_table3.p_target, pos_label=1)
        dlpyf1_score1_1 = f1_score('target', 'p_target', pos_label=1, castable=self.class_table3)
        
        self.assertAlmostEqual(skf1_score1, dlpyf1_score1)
        self.assertAlmostEqual(dlpyf1_score1, dlpyf1_score1_1)
        
        skf1_score2 = skf1(self.local_class3.target, self.local_class4.p_target, pos_label=1)    
        dlpyf1_score2 = f1_score(self.class_table3.target, self.class_table4.p_target, pos_label=1,
                                 id_vars=['id1', 'id2'])
        
        self.assertAlmostEqual(skf1_score2, dlpyf1_score2)
        
    def test_explained_variance_score(self):

        try:
            from sklearn.metrics import explained_variance_score as skevs
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")
        
        skevs_score1 = skevs(self.local_reg1.target, self.local_reg1.p_target)    
        dlpyevs_score1 = explained_variance_score('target', 'p_target', castable=self.reg_table1)
        
        self.assertAlmostEqual(skevs_score1, dlpyevs_score1)  
        
        skevs_score2 = skevs(self.local_reg1.target, self.local_reg2.p_target)    
        dlpyevs_score2 = explained_variance_score(self.reg_table1.target, self.reg_table2.p_target, 
                                                  id_vars='id1')
        
        self.assertAlmostEqual(skevs_score2, dlpyevs_score2)
        
    def test_mean_absolute_error(self):

        try:
            from sklearn.metrics import mean_absolute_error as skmae
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")
        
        skmae_score1 = skmae(self.local_reg1.target, self.local_reg1.p_target)    
        dlpymae_score1 = mean_absolute_error('target', 'p_target', castable=self.reg_table1)
        
        self.assertAlmostEqual(skmae_score1, dlpymae_score1)
        
        skmae_score2 = skmae(self.local_reg1.target, self.local_reg2.p_target)    
        dlpymae_score2 = mean_absolute_error(self.reg_table1.target, self.reg_table2.p_target,
                                             id_vars='id1')
        
        self.assertAlmostEqual(skmae_score2, dlpymae_score2)
        
    def test_mean_squared_error(self):

        try:
            from sklearn.metrics import mean_squared_error as skmse
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")
        
        skmse_score1 = skmse(self.local_reg1.target, self.local_reg1.p_target)    
        dlpymse_score1 = mean_squared_error('target', 'p_target', castable=self.reg_table1)
        
        self.assertAlmostEqual(skmse_score1, dlpymse_score1)
        
        skmse_score2 = skmse(self.local_reg1.target, self.local_reg2.p_target)    
        dlpymse_score2 = mean_squared_error(self.reg_table1.target,self.reg_table2.p_target,
                                            id_vars='id1')
        
        self.assertAlmostEqual(skmse_score2, dlpymse_score2)
        
    def test_mean_squared_log_error(self):

        try:
            from sklearn.metrics import mean_squared_log_error as skmsle
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skmsle_score1 = skmsle(self.local_reg1.target, self.local_reg1.p_target)    
        dlpymsle_score1 = mean_squared_log_error('target', 'p_target', castable=self.reg_table1)
        
        self.assertAlmostEqual(skmsle_score1, dlpymsle_score1)
        
        skmsle_score2 = skmsle(self.local_reg1.target, self.local_reg2.p_target)    
        dlpymsle_score2 = mean_squared_log_error(self.reg_table1.target, self.reg_table2.p_target,
                                                 id_vars='id1')
        dlpymsle_score2_1 = mean_squared_log_error(self.reg_table1.target, self.reg_table2.p_target)
        
        self.assertAlmostEqual(skmsle_score2, dlpymsle_score2)
        
    def test_r2_score(self):

        try:
            from sklearn.metrics import r2_score as skr2sc
        except:
            unittest.TestCase.skipTest(self, "sklearn is not found in the libraries")

        skr2sc_score1 = skr2sc(self.local_reg1.target, self.local_reg1.p_target)    
        dlpyr2sc_score1 = r2_score('target', 'p_target', castable=self.reg_table1)
        
        self.assertAlmostEqual(skr2sc_score1, dlpyr2sc_score1)
        
        skr2sc_score2 = skr2sc(self.local_reg1.target, self.local_reg2.p_target)    
        dlpyr2sc_score2 = r2_score(self.reg_table1.target, self.reg_table2.p_target,
                                   id_vars='id1')
        dlpyr2sc_score2_1 = r2_score(self.reg_table1.target, self.reg_table2.p_target)
        
        self.assertAlmostEqual(skr2sc_score2, dlpyr2sc_score2)
