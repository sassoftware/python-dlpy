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
import matplotlib.pyplot as plt
import numpy as np

def accuracy_score(castable, y_true, y_pred, normalize=True):
    '''
    Computes the classification accuracy score.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_pred : string
        The column name of the predicted class labels. 
    normalize : boolean, optional
        If ``False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.
        Default=True

    Returns
    -------
    score : float
        If ``normalize=False``, return the number of correctly classified samples.
        Otherwise, return the fraction of correctly classified samples.        

    '''
    
    matched_colname = 'matched'
    
    conn = castable.get_connection()
    
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
    
    return score


def confusion_matrix(castable, y_true, y_pred, labels=None):
    '''
    Computes the confusion matrix of a classification task.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_pred : string
        The column name of the predicted class labels. 
    labels : list, optional
        List of labels that can be used to reorder the matrix or 
        select the subset of the labels. If ``labels=None``, 
        all labels are included.
        Default=None

    Returns
    -------
    :class:`pandas.DataFrame`
        The column index is the predicted class labels. 
        The row index is the ground truth class labels.         

    '''
    
    res = castable.retrieve('crosstab', 
                             _messagelevel='error', 
                             row=y_true, col=y_pred)
    
    conf_mat = res['Crosstab']
    
    colinfo = castable.columninfo().ColumnInfo
    
    target_dtype = colinfo.Type[colinfo.Column==y_true].values[0]
    
    if target_dtype == 'double':
        target_index_dtype = np.float64
        conf_mat[y_true] = conf_mat[y_true].astype(target_index_dtype)
    elif target_dtype.startswith('int'):
        target_index_dtype = getattr(np, target_dtype)
        conf_mat[y_true] = conf_mat[y_true].astype(target_index_dtype)
        
    conf_mat.set_index(y_true, inplace=True)
    
    conf_mat.columns = conf_mat.index.copy()
    conf_mat.columns.name = y_pred
    
    if labels is None:
        return conf_mat
    else:
        if not isinstance(labels, list):
            labels = [labels]
        
        return conf_mat.loc[labels, labels]
    
def plot_roc(castable, y_true, y_score, pos_label, cutstep=0.001, 
             figsize=(8, 8), 
             fontsize_spec=None, linewidth=1):
    
    '''
    Plot the receiver operating characteristic (ROC) curve for binary classification 
    tasks.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_score : string
        The column name of estimated probability for the positive class. 
    pos_label : string, int or float
        The positive class label. 
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

    conn = castable.get_connection()
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
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
    
def plot_precision_recall(castable, y_true, y_score, pos_label, cutstep=0.001, 
                          figsize=(8, 8), fontsize_spec=None, linewidth=1):
    '''
    Plot the precision recall(PR) curve for binary classification 
    tasks.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_score : string
        The column name of estimated probability for the positive class. 
    pos_label : string, int or float
        The positive class label. 
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

    conn = castable.get_connection()
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    rocinfo = res['ROCInfo']
    
    rocinfo.loc[rocinfo.TP+ rocinfo.FP == 0, 'FDR'] = 0
    fdr = list(rocinfo.FDR) + [0]
    precision = [1-x for x in fdr]
    recall = list(rocinfo.Sensitivity) + [0]
        
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(recall, precision, linestyle='-', linewidth=linewidth)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_xlabel('Recall', fontsize=fontsize['xlabel'])
    ax.set_ylabel('Precision', fontsize=fontsize['ylabel'])
    ax.get_xaxis().set_tick_params(direction='out', labelsize=fontsize['xtick'])
    ax.get_yaxis().set_tick_params(direction='out', labelsize=fontsize['ytick'])
    ax.set_title('Precision-Recall curve', fontsize=fontsize['title'])
    
    return ax


def roc_auc_score(castable, y_true, y_score, pos_label, cutstep=0.001):
    '''
    Compute the area under the receiver operating characteristic (ROC) curve for binary classification 
    tasks.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_score : string
        The column name of estimated probability for the positive class. 
    pos_label : string, int or float
        The positive class label. 
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001.         

    Returns
    -------
    score : float

    '''
    
    if not isinstance(pos_label, str):
        pos_label = str(pos_label)

    conn = castable.get_connection()
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
    rocinfo = res['ROCInfo']
    
    auc_score = rocinfo.C.loc[0]
    
    return auc_score

def average_precision_score(castable, y_true, y_score, pos_label, cutstep=0.001, 
                            interpolate=False):
    '''
    Compute the average precision score for binary classification 
    tasks. 

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_score : string
        The column name of estimated probability for the positive class. 
    pos_label : string, int or float
        The positive class label. 
    cutstep : float > 0 and < 1, optional
        The stepsize of threshold cutoffs. 
        Default=0.001. 
    interpolate : boolean, optional
        If ``interpolate=True``, it is the area under the precision recall 
        curve with linear interpolation. Otherwise, it is defined as 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    Returns
    -------
    score : float

    '''

    if not isinstance(pos_label, str):
        pos_label = str(pos_label)

    conn = castable.get_connection()
    
    conn.retrieve('loadactionset', _messagelevel = 'error', actionset = 'percentile')
    
    res = conn.retrieve('percentile.assess', _messagelevel = 'error', 
                            table=castable, 
                            inputs=y_score, response=y_true, 
                            event=pos_label, cutstep=cutstep)
    
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

def f1_score(castable, y_true, y_pred, pos_label):
    '''
    Compute the f1 score of the binary classification task. f1 score is defined as
    :math:`\frac{2PR}{P+R}`, where :math:`P` is the precision and :math:`R` is 
    the recall. 

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth labels
    y_pred : string
        The column name of the predicted class labels.  
    pos_label : string, int or float
        The positive class label. 

    Returns
    -------
    score : float

    '''
    
    conf_mat = confusion_matrix(castable, y_true, y_pred)
    
    recall = conf_mat.loc[pos_label, pos_label]/conf_mat.loc[pos_label, :].sum()
    
    precision = conf_mat.loc[pos_label, pos_label]/conf_mat.loc[:, pos_label].sum()
    
    f1 = 2*precision*recall/(precision + recall)
    
    return f1


def explained_variance_score(castable, y_true, y_pred):
    '''
    Compute the explained variance score for a regression task. It is the 
    fraction of the target variable variance that is explained by the model.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth target values.
    y_pred : string
        The column name of the predicted target values. 

    Returns
    -------
    score : float

    '''
    
    error_colname = 'err'
    
    conn = castable.get_connection()
    
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

    return expl_var    

def mean_absolute_error(castable, y_true, y_pred):
    '''
    Compute the mean absolute error of a regression task.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth target values.
    y_pred : string
        The column name of the predicted target values. 

    Returns
    -------
    loss : float

    '''
    
    error_colname = 'abserr'
    
    conn = castable.get_connection()
    
    while error_colname in castable.columns:
        error_colname = random_name(name='abserr_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=abs({0}-{1})'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    mae = castable[error_colname].mean()

    return mae   

def mean_squared_error(castable, y_true, y_pred):
    '''
    Compute the mean squared error of a regression task.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth target values.
    y_pred : string
        The column name of the predicted target values. 

    Returns
    -------
    loss : float

    '''
    
    error_colname = 'err2'
    
    conn = castable.get_connection()
    
    while error_colname in castable.columns:
        error_colname = random_name(name='err2_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=({0}-{1})**2'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    mse = castable[error_colname].mean()

    return mse   

def mean_squared_log_error(castable, y_true, y_pred):
    '''
    Compute the mean squared logarithmic error of the regression tasks.

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth target values.
    y_pred : string
        The column name of the predicted target values. 

    Returns
    -------
    loss : float

    '''
    
    error_colname = 'logerr2'
    
    conn = castable.get_connection()
    
    while error_colname in castable.columns:
        error_colname = random_name(name='logerr2_')  
        
    castbl_params = {}
    castbl_params['computedvars'] = [{"name":error_colname}]
    code = '{2}=(log(1+{0})-log(1+{1}))**2'.format(y_true, y_pred, error_colname)
    castbl_params['computedvarsprogram'] = code
    castable = conn.CASTable(castable.name, **castbl_params)
    
    logerr2 = castable[error_colname].mean()

    return logerr2 


def r2_score(castable, y_true, y_pred):
    '''
    Compute the :math:`R^2` (coefficient of determination) regression score. 

    Parameters
    ----------
    castable : :class:`CASTable`
        The CASTable object to use as the source.
    y_true : string
        The column name of the ground truth target values.
    y_pred : string
        The column name of the predicted target values. 

    Returns
    -------
    loss : float

    '''
    
    mse = mean_squared_error(castable, y_true, y_pred)
    
    nobs = castable[[y_true, y_pred]].dropna().shape[0]
    
    sse = nobs*mse
    
    tss = castable[y_true].var()*(nobs-1)
    
    r2  = 1- sse/tss

    return r2  

   