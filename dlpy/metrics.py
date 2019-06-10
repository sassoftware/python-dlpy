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

''' Evaluation metrics for classification and regression tasks '''


from .utils import random_name
from swat.cas.table import CASColumn
from swat.cas.table import CASTable
import matplotlib.pyplot as plt
import numpy as np
import warnings

def accuracy_score(y_true, y_pred, castable=None, normalize=True, id_vars=None):
    '''
    Computes the classification accuracy score.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted class labels. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    normalize : boolean, optional
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
        Default = True
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment. 
        Default = None

    Returns
    -------
    score : float
        If ``normalize=False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.        

    '''

    
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
       
    matched_colname = 'matched'
    # check whether matched_colname is already in the castable, 
    # to avoid duplication or overwrite when creating computedvars. 
    while matched_colname in castable.columns:
        matched_colname = random_name(name='matched_')  
     
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":matched_colname}]
    code = 'if {0}={1} then {2}=1;else {2}=0'.format(y_true, y_pred, matched_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    if normalize:
        score = castable[matched_colname].mean()
    else:
        score = castable[matched_colname].sum()
    
    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
    
    return score


def confusion_matrix(y_true, y_pred, castable=None, labels=None, id_vars=None):
    '''
    Computes the confusion matrix of a classification task.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted class labels. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    labels : list, optional
        List of labels that can be used to reorder the matrix or 
        select the subset of the labels. If ``labels=None``, 
        all labels are included.
        Default=None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment. 
        Default = None


    Returns
    -------
    :class:`pandas.DataFrame`
        The column index is the predicted class labels. 
        The row index is the ground truth class labels.         

    '''
    
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=True, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    target_dtype = check_results[5]
    
    res = castable.retrieve('crosstab', 
                             _messagelevel='error', 
                             row=y_true, col=y_pred)
    
    conf_mat = res['Crosstab']
    
    if target_dtype == 'double':
        target_index_dtype = np.float64
        conf_mat[y_true] = conf_mat[y_true].astype(target_index_dtype)
    elif target_dtype.startswith('int'):
        target_index_dtype = getattr(np, target_dtype)
        conf_mat[y_true] = conf_mat[y_true].astype(target_index_dtype)
        
    conf_mat.set_index(y_true, inplace=True)
    
    conf_mat.columns = conf_mat.index.copy()
    conf_mat.columns.name = y_pred
    
    if tmp_table_created: # if tmp_table_created, tbl_name referes to the temporary table name
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
    
    if labels is None:
        return conf_mat
    else:
        if not isinstance(labels, list):
            labels = [labels]
        
        return conf_mat.loc[labels, labels]
    
def plot_roc(y_true, y_score, pos_label, castable=None, cutstep=0.001, 
             figsize=(8, 8), fontsize_spec=None, linewidth=1, id_vars=None):
    
    '''
    Plot the receiver operating characteristic (ROC) curve for binary classification 
    tasks.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_score has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_score has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    y_score : string or :class:`CASColumn`
        The column of estimated probability for the positive class. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    pos_label : string, int or float
        The positive class label. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_score and y_true are strings.
        Default = None 
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001. 
    figsize : tuple, optional
        The size of the generated figure.
        Default=(8, 8).
    fontsize_spec : dict, optional
        It specifies the fontsize for 'xlabel', 'ylabel', 'xtick', 'ytick' 
        and 'title'. (e.g. {'xlabel':14, 'ylabel':14}).
        If None, it will take the default fontsize, which are
        {'xlabel':16, 'ylabel':16, 'xtick':14, 'ytick':14, 'title':20}
        Default=None. 
    linewidth : float, optional
        It specify the line width for the ROC curve. 
        Default=1. 
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_score if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_score appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.
        

    Returns
    -------
    :class:`matplotlib.axes.Axes`   
    The x-axis is the false positive rate and the y-axis is the true positive rate.       

    '''
    
    
    fontsize = {'xlabel':16, 'ylabel':16, 'xtick':14,
                         'ytick':14, 'title':20}
    
    if fontsize_spec is not None:
        fontsize.update(fontsize_spec)
        
    if not isinstance(pos_label, str):
        pos_label = str(pos_label)
        
    check_results = _check_inputs(y_true, y_score, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_score = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    if tmp_table_created: # if tmp_tbl_created, tbl_name referes to the temporary table name
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
        
    rocinfo = res['ROCInfo']

    fpr = list(rocinfo.FPR) + [0]
    tpr = list(rocinfo.Sensitivity) + [0]
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(fpr, tpr, linestyle='-', linewidth=linewidth)
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.plot([0,1], [0,1], linestyle='--', linewidth=linewidth)
    ax.set_xlabel('False Positive Rate', fontsize=fontsize['xlabel'])
    ax.set_ylabel('True Positive Rate', fontsize=fontsize['ylabel'])
    ax.get_xaxis().set_tick_params(direction='out', labelsize=fontsize['xtick'])
    ax.get_yaxis().set_tick_params(direction='out', labelsize=fontsize['ytick'])
    ax.set_title('ROC curve', fontsize=fontsize['title'])
    
    return ax
    
def plot_precision_recall(y_true, y_score, pos_label, castable=None, cutstep=0.001, 
                          figsize=(8, 8), fontsize_spec=None, linewidth=1, id_vars=None):
    '''
    Plot the precision recall(PR) curve for binary classification 
    tasks.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_score has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_score has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    y_score : string or :class:`CASColumn`
        The column of estimated probability for the positive class. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    pos_label : string, int or float
        The positive class label. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_score and y_true are strings.
        Default = None 
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001. 
    figsize : tuple, optional
        The size of the generated figure.
        Default=(8, 8).
    fontsize_spec : dict, optional
        It specifies the fontsize for 'xlabel', 'ylabel', 'xtick', 'ytick' 
        and 'title'. (e.g. {'xlabel':14, 'ylabel':14}).
        If None, it will take the default fontsize, which are
        {'xlabel':16, 'ylabel':16, 'xtick':14, 'ytick':14, 'title':20}
        Default=None. 
    linewidth : float, optional
        It specify the line width for the ROC curve. 
        Default=1.  
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_score if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_score appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.

    Returns
    -------
    :class:`matplotlib.axes.Axes`
    The x-axis is the recall(sensitivity) and the y-axis is the precision.       

    '''
    
    
    fontsize = {'xlabel':16, 'ylabel':16, 'xtick':14,
                         'ytick':14, 'title':20}
    
    if fontsize_spec is not None:
        fontsize.update(fontsize_spec)
        
    if not isinstance(pos_label, str):
        pos_label = str(pos_label)

    check_results = _check_inputs(y_true, y_score, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_score = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    if tmp_table_created: # if tmp_tbl_created, tbl_name referes to the temporary table name
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
    
    rocinfo = res['ROCInfo']
    
    rocinfo.loc[rocinfo.TP+ rocinfo.FP == 0, 'FDR'] = 0
    fdr = list(rocinfo.FDR) + [0]
    precision = [1-x for x in fdr]
    recall = list(rocinfo.Sensitivity) + [0]
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(recall, precision, linestyle='-', linewidth=linewidth)
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlim([-0.01, 1.01])
    ax.set_xlabel('Recall', fontsize=fontsize['xlabel'])
    ax.set_ylabel('Precision', fontsize=fontsize['ylabel'])
    ax.get_xaxis().set_tick_params(direction='out', labelsize=fontsize['xtick'])
    ax.get_yaxis().set_tick_params(direction='out', labelsize=fontsize['ytick'])
    ax.set_title('Precision-Recall curve', fontsize=fontsize['title'])
    
    return ax


def roc_auc_score(y_true, y_score, pos_label, castable=None, cutstep=0.001, id_vars=None):
    '''
    Compute the area under the receiver operating characteristic (ROC) curve for binary classification 
    tasks.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_score has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_score has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    y_score : string or :class:`CASColumn`
        The column of estimated probability for the positive class. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    pos_label : string, int or float
        The positive class label. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_score and y_true are strings.
        Default = None 
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001. 
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_score if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_score appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.
        

    Returns
    -------
    score : float

    '''
    
    if not isinstance(pos_label, str):
        pos_label = str(pos_label)

    check_results = _check_inputs(y_true, y_score, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_score = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    if tmp_table_created: # if tmp_tbl_created, tbl_name referes to the temporary table name
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
    
    rocinfo = res['ROCInfo']
    
    auc_score = rocinfo.C.loc[0]
    
    return auc_score

def average_precision_score(y_true, y_score, pos_label, castable=None, cutstep=0.001, 
                            interpolate=False, id_vars=None):
    '''
    Compute the average precision score for binary classification 
    tasks. 

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_score has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_score has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    y_score : string or :class:`CASColumn`
        The column of estimated probability for the positive class. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_score 
        and y_true are :class:`CASColumn`, they can be in different CASTable. 
    pos_label : string, int or float
        The positive class label. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_score and y_true are strings.
        Default = None  
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001. 
    interpolate : boolean, optional
        If ``interpolate=True``, it is the area under the precision recall 
        curve with linear interpolation. Otherwise, it is defined as 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_score if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_score appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.


    Returns
    -------
    score : float

    '''

    if not isinstance(pos_label, str):
        pos_label = str(pos_label)

    check_results = _check_inputs(y_true, y_score, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_score = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    if tmp_table_created: # if tmp_tbl_created, tbl_name referes to the temporary table name
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 
    
    rocinfo = res['ROCInfo']
    
    rocinfo.loc[rocinfo.TP+ rocinfo.FP == 0, 'FDR'] = 0
    fdr = list(rocinfo.FDR) + [0]
    precision = [1-x for x in fdr]
    recall = list(rocinfo.Sensitivity) + [0]
    
    if interpolate:
        #Calculate the area under the PR curve using trapezoidal rule, with linear interpolation
        ap = sum([np.mean(precision[i:i+2])*(recall[i]-recall[i+1]) 
        for i in range(len(recall)-1)])
    else:
        #Use the formulation same as scikit-learn without linear interpolation.
        #https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
        ap = sum([precision[i]*(recall[i]-recall[i+1]) 
        for i in range(len(recall)-1)])        
    
    return ap    

def f1_score(y_true, y_pred, pos_label, castable=None, id_vars=None):
    '''
    Compute the f1 score of the binary classification task. f1 score is defined as
    :math:`\frac{2PR}{P+R}`, where :math:`P` is the precision and :math:`R` is 
    the recall. 

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted class labels. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    pos_label : string, int or float
        The positive class label. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None 
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.



    Returns
    -------
    score : float

    '''
    
    conf_mat = confusion_matrix(y_true, y_pred, castable=castable, id_vars=id_vars)
    
    recall = conf_mat.loc[pos_label, pos_label]/conf_mat.loc[pos_label, :].sum()
    
    precision = conf_mat.loc[pos_label, pos_label]/conf_mat.loc[:, pos_label].sum()
    
    f1 = 2*precision*recall/(precision + recall)
    
    return f1


def explained_variance_score(y_true, y_pred, castable=None, id_vars=None):
    '''
    Compute the explained variance score for a regression task. It is the 
    fraction of the target variable variance that is explained by the model.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth target values. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted target values. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.

    Returns
    -------
    score : float

    '''
    
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    error_colname = 'err'
    # check whether error_colname is already in the castable, 
    # to avoid duplication or overwrite when creating computedvars. 
    while error_colname in castable.columns:
        error_colname = random_name(name='err_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}={0}-{1}'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    total_var = castable[y_true].var()
    err_var   = castable[error_colname].var()
    
    expl_var  = 1 - err_var/total_var
    
    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 

    return expl_var    

def mean_absolute_error(y_true, y_pred, castable=None, id_vars=None):
    '''
    Compute the mean absolute error of a regression task.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth target values. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted target values. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.
        
    Returns
    -------
    loss : float

    '''
  
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]

    error_colname = 'abserr'
    # check whether error_colname is already in the castable, 
    # to avoid duplication or overwrite when creating computedvars.     
    while error_colname in castable.columns:
        error_colname = random_name(name='abserr_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=abs({0}-{1})'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    mae = castable[error_colname].mean()
    
    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 

    return mae   

def mean_squared_error(y_true, y_pred, castable=None, id_vars=None):
    '''
    Compute the mean squared error of a regression task.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth target values. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted target values. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.

    Returns
    -------
    loss : float

    '''
   
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]

    error_colname = 'err2'
    # check whether error_colname is already in the castable, 
    # to avoid duplication or overwrite when creating computedvars.     
    while error_colname in castable.columns:
        error_colname = random_name(name='err2_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=({0}-{1})**2'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    mse = castable[error_colname].mean()

    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 

    return mse   

def mean_squared_log_error(y_true, y_pred, castable=None, id_vars=None):
    '''
    Compute the mean squared logarithmic error of the regression tasks.

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth target values. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted target values. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.

    Returns
    -------
    loss : float

    '''
    
    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]

    error_colname = 'logerr2'
    # check whether error_colname is already in the castable, 
    # to avoid duplication or overwrite when creating computedvars.     
    while error_colname in castable.columns:
        error_colname = random_name(name='logerr2_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=(log(1+{0})-log(1+{1}))**2'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    logerr2 = castable[error_colname].mean()

    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name) 

    return logerr2 


def r2_score(y_true, y_pred, castable=None, id_vars=None):
    '''
    Compute the R^2 (coefficient of determination) regression score. 

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth target values. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted target values. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.

    Returns
    -------
    loss : float

    '''
    
    mse = mean_squared_error(y_true, y_pred, castable=castable, id_vars=id_vars)

    check_results = _check_inputs(y_true, y_pred, castable=castable, 
                                  return_target_dtype=False, id_vars=id_vars)
    y_true = check_results[0]
    y_pred = check_results[1]
    castable = check_results[2]
    conn = check_results[3]
    tmp_table_created = check_results[4]
    
    nobs = castable[[y_true, y_pred]].dropna().shape[0]
    
    sse = nobs*mse
    
    tss = castable[y_true].var()*(nobs-1)
    
    r2  = 1- sse/tss
    
    if tmp_table_created:  # if tmp_table_created, tbl_name referes to the temporary table name   
        conn.retrieve('table.droptable', _messagelevel='error', name=castable.name)     

    return r2  


def _check_inputs(y_true, y_pred, castable=None, return_target_dtype=False, id_vars=None):
    '''
    Check the input argument y_true, y_pred, and return their names if they are CASColumn.
    If y_true, and y_pred is in the form of CASColumn and from different CASTables, 
    a temporary CASTable will be created which contains both columns. 

    Parameters
    ----------
    y_true : string or :class:`CASColumn`
        The column of the ground truth labels. If it is a string, then 
        y_pred has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_pred has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    y_pred : string or :class:`CASColumn`
        The column of the predicted class labels. If it is a string, then 
        y_true has to be a string and they both belongs to the same CASTable specified 
        by the castable argument. If it is a :class:`CASColumn`, then y_true has to be 
        a :class:`CASColumn`, and the castable argument is ignored. When both y_pred 
        and y_true are :class:`CASColumn`, they can be in different CASTables. 
    castable : :class:`CASTable`, optional
        The CASTable object to use as the source if the y_pred and y_true are strings.
        Default = None
    return_target_dtype : boolean, optional
        If True, return the data type of y_true in the CASTable.
        Default = False
    id_vars : string or list of strings, optional
        Column names that serve as unique id for y_true and y_pred if they are 
        from different CASTables. The column names need to appear in both CASTables, 
        and they serve to match y_true and y_pred appropriately, since observation 
        orders can be shuffled in distributed computing environment.
        Default = None.
        
    Returns
    -------
    y_true : string
        The column name of the y_true column.
    y_pred : string
        The column name of the y_pred column.
    castable : :class:`CASTable`
        The original CASTable if y_true and y_pred are in the same castable. The 
        temporary table that contain both columns if y_true and y_pred is from 
        different CASTable.
    conn : :class:`CAS` 
        The connection on the CASColumn or CASTable.
    tmp_table_created : boolean
        Whether a temporary CASTable is created to host y_true and y_pred.
    target_dtype : string
        The data type of y_true in the CASTable. 
        Only provided if `return_target_dtype` is True.
    '''
    tmp_table_created = False
    
    if isinstance(y_pred, str) and isinstance(y_true, str):
        if not isinstance(castable, CASTable):
            raise ValueError('castable need to be a CASTable if y_true and y_pred are strings')
        conn = castable.get_connection()
        if return_target_dtype:
            colinfo = castable.columninfo().ColumnInfo
            target_dtype = colinfo.Type[colinfo.Column==y_true].values[0]
    elif isinstance(y_pred, CASColumn) and isinstance(y_true, CASColumn):
        conn = y_true.get_connection()
        y_true_tblname = y_true.to_outtable_params()['name']
        y_pred_tblname = y_pred.to_outtable_params()['name']
        if return_target_dtype:
            colinfo = y_true.columninfo().ColumnInfo
            target_dtype = colinfo.Type[colinfo.Column==y_true.name].values[0]            
        y_true = y_true.name
        y_pred = y_pred.name
        if y_true_tblname != y_pred_tblname:
            tmp_table_name = random_name('metric_tmp',6)
            if id_vars is None:
                warnings.warn('{} and {} are from different CASTables, '.format(y_true, y_pred) + 
                              'and their appropriate matching may not be guaranteed '+
                              'unless id_vars argument is provided.')
                sascode = '''
                data {};
                merge {}(keep={}) {}(keep={});
                run;
                '''.format(tmp_table_name, y_true_tblname, y_true, y_pred_tblname, y_pred) 
                
                conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode, single='Yes')
            else:
                if not isinstance(id_vars, list):
                    id_vars = [id_vars]
                y_true_keep = ' '.join([y_true]+id_vars)
                y_pred_keep = ' '.join([y_pred]+id_vars)
                by_var = ' '.join(id_vars)
                sascode = '''
                data {};
                merge {}(keep={}) {}(keep={});
                by {};
                run;
                '''.format(tmp_table_name, y_true_tblname, y_true_keep, y_pred_tblname, y_pred_keep, by_var) 
                
                conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)                

            castable = conn.CASTable(tmp_table_name)
            tmp_table_created = True
        else:
            castable = conn.CASTable(y_true_tblname)

            
    else:
        raise ValueError('Input for ground truth and predicted value need to be the same type of either '+
                         'strings representing column names or CASColumns')    
    
    
    if return_target_dtype:
        return (y_true, y_pred, castable, conn, tmp_table_created, target_dtype)
    else:
        return (y_true, y_pred, castable, conn, tmp_table_created)
    
    
    
    
    
