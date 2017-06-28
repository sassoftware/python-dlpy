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

'''
Model object for deep learning.
'''


from swat import *
from .utils import random_name



class Model:
    '''
    Model

    Parameters:

    ----------
    sess :
        Specifies the session of the CAS connection.
    model_name : string
        Specifies the name of the deep learning model.
    model_weights : string, dictionary or CAS table, optional
        Specify the weights of the deep learning model.
        If not specified, random initial will be used.
        Default : None

    Returns

    -------
    A deep learning model objects.
    '''

    def __init__(self, sess, model_name, model_weights=None):

        if not sess.queryactionset('deepLearn')['deepLearn']:
            sess.loadactionset('deepLearn')

        # self.table = sess.CASTable(model_name)
        self.model_name = model_name
        self.sess = sess
        if model_weights is None:
            self.model_weights = None
        else:
            self.set_weights(model_weights)
        self.valid_res = None
        self.valid_feature_maps = None
        self.valid_conf_mat = None

    def load(self, path):

        sess = self.sess
        caslibname = random_name('Caslib', 6)
        castblname = random_name('Table', 6)
        sess.addCaslib(name=caslibname, path=path, activeOnAdd=False)

        sess.image.saveImages(caslib=caslibname,
                              images=vl(table=dict(name=self.tbl['name'])),
                              labelLevels=1)
        sess.dropcaslib(caslib=caslibname)

    def model_info(self):
        return self.sess.modelinfo(modelTable=self.model_name)

    def set_weights(self, model_weights):
        if isinstance(model_weights, str):
            self.model_weights = self.sess.CASTable(model_weights)
        elif isinstance(model_weights, dict):
            self.model_weights = self.sess.CASTable(**model_weights)
        else:
            self.model_weights = model_weights

    def fit(self, trainTbl, inputs='_image_', target='_label_',
            miniBatchSize=1, maxEpochs=5, logLevel=3, optimizer=None,
            **kwargs):

        '''
        Fit function of the deep learning model.

        Parameters:

        ----------
        trainTbl : CAS table
            Specify the training data for the model.
        inputs : string, optional
            Specify the variable name of in the trainTbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the trainTbl, that is the response of the deep learning model.
            Default : '_label_'.
        miniBatchSize : integer, optional
            Specify the number of observations per thread in a mini-batch..
            Default : 1.
        maxEpochs : int64, optional
            Specify the maximum number of epochs.
            Default : 5.
        logLevel : int 0-3, optional
            Specify  how progress messages are sent to the client.
                0 : no messages are sent.
                1 : send the start and end messages.
                2 : send the iteration history for each epoch.
                3 : send the iteration history for each batch.
            Default : 3.
        optimizer: dictionary, optional
            Specify the options for the optimizer in the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_dlcommon_dlOptimizerOpts.html#type_synchronous
            for detail.
        kwargs: dictionary, optional
            Specify the optional arguments for the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlTrain.html
            for detail.

        Returns

        ----------
        Retrun a fetch result to the client, about the trained model summary.
        The updated model weights are automatically assigned to the Model object.

        '''

        if optimizer is None:
            optimizer = dict(algorithm=dict(method='momentum',
                                            clipgradmin=-1000,
                                            clipgradmax=1000,
                                            learningRate=0.0001,
                                            lrpolicy='step',
                                            stepsize=15,
                                            ),
                             miniBatchSize=miniBatchSize,
                             maxEpochs=maxEpochs,
                             regL2=0.0005,
                             logLevel=logLevel
                             # ,snapshotfreq=10
                             )
        else:
            key_args = ['miniBatchSize', 'maxEpochs', 'logLevel']
            for key_arg in key_args:
                if optimizer[key_arg] is None:
                    optimizer[key_arg] = eval(key_arg)

        r = self.sess.dltrain(model=self.model_name,
                              table=trainTbl,
                              inputs=inputs,
                              target=target,
                              initWeights=self.model_weights,
                              modelWeights=dict(name='_model_weights_', replace=True),
                              optimizer=optimizer,
                              **kwargs)

        self.set_weights('_model_weights_')
        return r

    def predict(self, validTbl, inputs='_image_', target='_label_',
                layer_output=False, **kwargs):
        '''
        Function of scoring the deep learning model on a validation data set.

        Parameters:

        ----------
        validTbl : CAS table
            Specify the validating data for the model.
        inputs : string, optional
            Specify the variable name of in the validTbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the validTbl, that is the response of the deep learning model.
            Default : '_label_'.
        layer_output : boolean, optional
            Specify whether to output the feather maps of the layers.
        kwargs: dictionary, optional
            Specify the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        Retrun a fetch result to the client, about the validation results.
        The validation results are automatically assigned to the Model object.
        '''

        if layer_output:

            res = self.sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                                    table=validTbl,
                                    copyVars=[inputs, target],
                                    layerOut=dict(name='Feature_maps', replace=True),
                                    layerImageType='jpg',
                                    casout=dict(name='valid_res', replace=True),
                                    **kwargs)

            self.valid_res = self.sess.CASTable('valid_res')
            self.valid_feature_maps = self.sess.CASTable('Feature_maps')
            self.valid_conf_mat = self.sess.crosstab(
                table='valid_res', row='_label_', col='_dl_predname_')
            return res, self.valid_conf_mat

        else:
            res = self.sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                                    table=validTbl,
                                    copyVars=[inputs, target],
                                    casout=dict(name='valid_res', replace=True),
                                    **kwargs)

            self.valid_res = self.sess.CASTable('valid_res')
            self.valid_conf_mat = self.sess.crosstab(
                table='valid_res', row='_label_', col='_dl_predname_')
            return res, self.valid_conf_mat
