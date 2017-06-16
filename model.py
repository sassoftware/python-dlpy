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


class Model:
    '''
    Model

    Parameters:

    ----------
    sess :
        Specify the session of the CAS connection.
    model_name : string
        Specify the name of the deep learning model.
    model_weights : string, dictionary or CAS table, optional
        Specify the weights of the deep learning model.
        Default = None
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
            optimizer=None, dltrain_args=None):
        '''
        Fit function of the deep learning model.

        Parameters:

        ----------
        trainTbl : CAS table
            Specify the training data for the model.
        inputs : string, optional
            Specify the variable name of in the trainTbl, that is the input of the deep learning model.
            Default = '_image_'.
        target : string, optional
            Specify the variable name of in the trainTbl, that is the response of the deep learning model.
            Default = '_label_'.
        optimizer: dictionary, optional
            Specify the options for the optimizer in the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_dlcommon_dlOptimizerOpts.html#type_synchronous
            for detail.
        dltrain_args: dictionary, optional
            Specify the optional arguments for the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlTrain.html
            for detail.
        '''

        if optimizer is None:
            optimizer = dict(mode=dict(type='synchronous'),
                             algorithm=dict(method='momentum',
                                            clipgradmin=-1000,
                                            clipgradmax=1000,
                                            learningRate=0.0001,
                                            lrpolicy='step',
                                            stepsize=15,
                                            # uselocking=False,
                                            ),
                             miniBatchSize=1,
                             maxEpochs=1,
                             regL2=0.0005,
                             logLevel=3
                             # ,snapshotfreq=10
                             )
        if dltrain_args is None:
            dltrain_args = dict(nthreads=12, seed=55020, targetOrder='ascending')
        r = self.sess.dltrain(model=self.model_name,
                              table=trainTbl,
                              inputs=inputs,
                              target=target,
                              initWeights=self.model_weights,
                              modelWeights=dict(name='_model_weights_', replace=True),
                              optimizer=optimizer,
                              **dltrain_args
                              )
        self.set_weights('_model_weights_')
        return r

    def predict(self, validTbl, layer_output=False):
        if layer_output:
            res = self.sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                                    table=validTbl,
                                    copyVars=['_id_', '_path_', '_label_'],
                                    layerOut=dict(name='Feature_maps', replace=True),
                                    layerImageType='jpg',
                                    minibatchSize=1,
                                    casout=dict(name='valid_res', replace=True))
            self.valid_res = self.sess.CASTable('valid_res')
            self.valid_feature_maps = self.sess.CASTable('Feature_maps')
            self.valid_conf_mat = self.sess.crosstab(
                table='valid_res', row='_label_', col='_dl_predname_')
            return res
        else:
            res = self.sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                                    table=validTbl,
                                    copyVars=['_id_', '_path_', '_label_'],
                                    minibatchSize=1,
                                    casout=dict(name='valid_res', replace=True)
                                    )
            self.valid_res = self.sess.CASTable('valid_res')
            self.valid_conf_mat = self.sess.crosstab(
                table='valid_res', row='_label_', col='_dl_predname_')
            return res
