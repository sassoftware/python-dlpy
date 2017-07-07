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

from .utils import random_name
from .utils import input_table_check



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

    def __init__(self, sess, model_name=None, model_weights=None):

        if not sess.queryactionset('deepLearn')['deepLearn']:
            sess.loadactionset('deepLearn')

        # self.table = sess.CASTable(model_name)
        self.sess = sess
        if model_weights is None:
            self.model_weights = None
        else:
            self.set_weights(model_weights)

        if model_name is None:
            self.model_name = random_name('Model', 6)
        elif type(model_name) is not str:
            raise TypeError('model_name has to be a string type.')
        else:
            self.model_name = model_name

        self.valid_res = None
        self.valid_feature_maps = None
        self.valid_conf_mat = None

    def load(self, path):

        sess = self.sess

        dir_name, file_name = path.rsplit('/', 1)

        CAS_LibName = random_name('Caslib', 6)
        CAS_TblName = file_name.split('.')[0]
        sess.addCaslib(name=CAS_LibName, path=dir_name, activeOnAdd=False)

        sess.loadtable(caslib=CAS_LibName, path=file_name, casout=CAS_TblName)
        model_name = sess.fetch(table=dict(name=CAS_TblName,
                                           where='_DLKey1_= "modeltype"')).Fetch._DLKey0_[0]
        self.model_name = model_name

        if model_name is not CAS_TblName:
            sess.partition(casout=model_name, table=CAS_TblName)
            sess.droptable(table=CAS_TblName)
        sess.dropcaslib(caslib=CAS_LibName)

    def model_info(self):
        return self.sess.modelinfo(modelTable=self.model_name)

    def set_weights(self, model_weights):
        sess = self.sess
        if isinstance(model_weights, str):
            self.model_weights = sess.CASTable(model_weights)
        elif isinstance(model_weights, dict):
            self.model_weights = sess.CASTable(**model_weights)
        else:
            self.model_weights = model_weights

    def fit(self, input_table, inputs='_image_', target='_label_',
            miniBatchSize=1, maxEpochs=5, logLevel=3, optimizer=None,
            **kwargs):

        '''
        Fit function of the deep learning model.

        Parameters:

        ----------
        input_table : A CAS table object, a string specifies the name of the CAS table,
                   a dictionary specifies the CAS table, or an Image object.
            Specify the training data for the model.
        inputs : string, optional
            Specify the variable name of in the input_table, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the input_table, that is the response of the deep learning model.
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
        sess = self.sess
        input_table = input_table_check(input_table)

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

        r = sess.dltrain(model=self.model_name,
                         table=input_table,
                         inputs=inputs,
                         target=target,
                         initWeights=self.model_weights,
                         modelWeights=dict(name='_model_weights_', replace=True),
                         optimizer=optimizer,
                         **kwargs)

        self.set_weights('_model_weights_')
        self.training_history = r.OptIterHistory

        return r

    def plot_training_history(self):
        self.training_history[['Loss', 'FitError']].plot(figsize=(12, 5))

    def predict(self, input_table, inputs='_image_', target='_label_',
                output_feature_maps=None, **kwargs):
        '''
        Function of scoring the deep learning model on a validation data set.

        Parameters:

        ----------
        input_table : CAS table
            Specify the validating data for the model.
        inputs : string, optional
            Specify the variable name of in the input_table, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the input_table, that is the response of the deep learning model.
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
        sess = self.sess
        input_table = input_table_check(input_table)

        valid_res_tbl = random_name('Valid_Res')

        if output_feature_maps:
            feature_maps_table = random_name('Feature_map')
            res = sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                               table=input_table,
                               copyVars=[inputs, target],
                               layerOut=dict(name=feature_maps_table, replace=True),
                               layerImageType='jpg',
                               layerList=output_feature_maps,
                               casout=dict(name=valid_res_tbl, replace=True),
                               **kwargs)

            self.valid_feature_maps = sess.CASTable(valid_res_tbl)


        else:
            res = sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                               table=input_table,
                               copyVars=[inputs, target],
                               casout=dict(name='valid_res', replace=True),
                               **kwargs)

        self.valid_res = sess.CASTable(valid_res_tbl)
        self.valid_score = res.ScoreInfo
        self.valid_conf_mat = sess.crosstab(
            table=valid_res_tbl, row=target, col='_dl_predname_')

    def save_to_astore(self, filename):
        sess = self.sess

        if not sess.queryactionset('astore')['astore']:
            sess.loadactionset('astore')
        CAS_TblName = random_name('Model_astore')
        sess.dlexportmodel(casout=CAS_TblName,
                           initWeights=self.model_weights,
                           modelTable=self.model_name)
        model_astore = sess.download(rstore=CAS_TblName)
        with open(filename, 'wb') as file:
            file.write(model_astore['blob'])

    def save_to_table(self, model_tbl_file, weight_tbl_file=None, attr_tbl_file=None):
        sess = self.sess

        sess.table.save(table=self.model_name,
                        name=model_tbl_file,
                        replace=True, caslib='CASUSER')
        if weight_tbl_file is not None:
            sess.table.save(table=self.model_weights,
                            name=weight_tbl_file,
                            replace=True, caslib='CASUSER')
            if attr_tbl_file is not None:
                CAS_TblName = random_name('Attr_Tbl')
                sess.table.attribute(caslib='CASUSER', task='CONVERT', name=self.model_weights,
                                     attrtable=CAS_TblName)
                sess.table.save(table=CAS_TblName,
                                name=attr_tbl_file,
                                replace=True, caslib='CASUSER')
