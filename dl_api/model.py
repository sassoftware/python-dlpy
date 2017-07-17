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
import numpy as np
import matplotlib.pyplot as plt


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
        self.n_epoch = None
        self.training_history = None

    def load(self, path):
        '''
        Function to load the deep learning model architecture from existing table.

        Parameters:

        ----------
        path: str
            Specify the full path of the table.
            Note: the path need to be in Linux path format.
        '''
        sess = self.sess

        dir_name, file_name = path.rsplit('/', 1)

        CAS_lib_name = random_name('Caslib', 6)
        CAS_tbl_name = file_name.split('.')[0]
        sess.addCaslib(name=CAS_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))
        sess.loadtable(caslib=CAS_lib_name, path=file_name, casout=CAS_tbl_name)
        model_name = sess.fetch(table=dict(name=CAS_tbl_name,
                                           where='_DLKey1_= "modeltype"')).Fetch['_DLKey0_'][0]
        self.model_name = model_name

        if model_name.lower() != CAS_tbl_name.lower():
            sess.partition(casout=model_name, table=CAS_tbl_name)
            sess.droptable(table=CAS_tbl_name)
        sess.dropcaslib(caslib=CAS_lib_name)

    def model_info(self):
        '''
        Function to return the information of the model table.
        '''
        return self.sess.modelinfo(modelTable=self.model_name)

    def set_weights(self, weight_tbl):

        '''
        Function to assign the weight to the model.


        Parameters:

        ----------
        weight_tbl : A CAS table object, a string specifies the name of the CAS table,
                   a dictionary specifies the CAS table.
            Specify the weights for the model.

        '''

        sess = self.sess

        weight_tbl = input_table_check(weight_tbl)
        self.model_weights = sess.CASTable(**weight_tbl)

    def fit(self, data, inputs='_image_', target='_label_',
            miniBatchSize=1, maxEpochs=5, logLevel=3, lr=0.01,
            optimizer=None,
            **kwargs):

        '''
        Train the deep learning model based the training data in the input table.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                   a dictionary specifies the CAS table, or an Image object.
            Specify the training data for the model.
        inputs : string, optional
            Specify the variable name of in the input_tbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the input_tbl, that is the response of the deep learning model.
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
        input_tbl = input_table_check(data)

        if optimizer is None:
            optimizer = dict(algorithm=dict(method='momentum',
                                            clipgradmin=-1000,
                                            clipgradmax=1000,
                                            learningRate=lr,
                                            lrpolicy='step',
                                            stepsize=15,
                                            ),
                             miniBatchSize=miniBatchSize,
                             maxEpochs=maxEpochs,
                             # regL2=0.0005,
                             logLevel=logLevel
                             # ,snapshotfreq=10
                             )
        else:
            key_args = ['miniBatchSize', 'maxEpochs', 'logLevel']
            for key_arg in key_args:
                if optimizer[key_arg] is None:
                    optimizer[key_arg] = eval(key_arg)

        r = sess.dltrain(model=self.model_name,
                         table=input_tbl,
                         inputs=inputs,
                         target=target,
                         initWeights=self.model_weights,
                         modelWeights=dict(name='_model_weights_', replace=True),
                         optimizer=optimizer,
                         **kwargs)

        if self.n_epoch is None:
            self.n_epoch = maxEpochs
            self.training_history = r.OptIterHistory
        else:
            temp = r.OptIterHistory
            temp.Epoch += self.n_epoch
            self.training_history = self.training_history.append(temp)
            self.n_epoch += maxEpochs
        self.training_history.index = range(0, self.n_epoch)
        self.set_weights('_model_weights_')

        return r

    def plot_training_history(self):
        '''
        Function to display the training iteration history.
        '''
        self.training_history[['Loss', 'FitError']].plot(figsize=(12, 5))

    def predict(self, input_tbl, inputs='_image_', target='_label_', **kwargs):
        '''
        Function of scoring the deep learning model on a validation data set.

        Parameters:

        ----------
        input_tbl : A CAS table object, a string specifies the name of the CAS table,
                      a dictionary specifies the CAS table, or an Image object.
            Specify the validating data for the prediction.
        inputs : string, optional
            Specify the variable name of in the input_tbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specify the variable name of in the input_tbl, that is the response of the deep learning model.
            Default : '_label_'.
        kwargs: dictionary, optional
            Specify the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.


        '''
        sess = self.sess
        input_tbl = input_table_check(input_tbl)

        valid_res_tbl = random_name('Valid_Res')

        res = sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                           table=input_tbl,
                           copyVars=[inputs, target],
                           randomFlip='NONE',
                           randomCrop='NONE',
                           casout=dict(name=valid_res_tbl, replace=True),
                           **kwargs)

        self.valid_res = sess.CASTable(valid_res_tbl)
        self.valid_score = res.ScoreInfo
        self.valid_conf_mat = sess.crosstab(
            table=valid_res_tbl, row=target, col='_dl_predname_')
        return self.valid_score

    def get_feature_maps(self, data, image_id=1, **kwargs):
        '''
        Function to extract the feature maps for a validation image.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
               a dictionary specifies the CAS table, or an Image object.
            Specify the table containing the image data.
        image_id : int, optional
            Specify the image id of the image.
            Default : 1.
        kwargs: dictionary, optional
            Specify the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        Retrun an instance variable of the Model object, which is a feature map object.
        '''

        sess = self.sess
        input_tbl = input_table_check(data)

        feature_maps_tbl = random_name('Feature_Maps') + '_{}'.format(image_id)
        print(feature_maps_tbl)
        sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                     table=dict(**input_tbl),
                     # , where='_id_={}'.format(image_id)),
                     layerOut=dict(name=feature_maps_tbl, replace=True),
                     randomFlip='NONE',
                     randomCrop='NONE',
                     layerImageType='jpg',
                     **kwargs)
        print('checked')
        layer_out_jpg = sess.CASTable(feature_maps_tbl)
        feature_maps_names = [i for i in layer_out_jpg.columninfo().ColumnInfo.Column]
        feature_maps_structure = dict()
        for feature_map_name in feature_maps_names:
            feature_maps_structure[int(feature_map_name.split('_')[2])] = int(feature_map_name.split('_')[4]) + 1

        self.valid_feature_maps = Feature_Maps(self.sess, feature_maps_tbl, structure=feature_maps_structure)

    def get_features(self, input_tbl, dense_layer, target='_label_', **kwargs):
        '''
        Function to extract the features for a data table.

        Parameters:

        ----------
        input_tbl : A CAS table object, a string specifies the name of the CAS table,
                    a dictionary specifies the CAS table, or an Image object.
            Specify the table containing the image data.
        dense_layer : str
            Specify the name of the layer that is extracted.
        target : str, optional
            Specify the name of the column including the response variable.
        kwargs: dictionary, optional
            Specify the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        X : ndarray of size n by p, where n is the sample size and p is the number of features.
            The features extracted by the model at the specified dense_layer.
        y : ndarray of size n.
            The response variable of the original data.
        '''

        sess = self.sess
        input_tbl = input_table_check(input_tbl)
        feature_tbl = random_name('Features')

        sess.dlscore(model=self.model_name, initWeights=self.model_weights,
                     table=dict(**input_tbl),
                     layerOut=dict(name=feature_tbl, replace=True),
                     layerList=dense_layer,
                     layerImageType='wide',
                     randomFlip='NONE',
                     randomCrop='NONE',
                     **kwargs)
        X = sess.CASTable(feature_tbl).as_matrix()
        y = sess.CASTable(**input_tbl)[target].as_matrix().ravel()

        return X, y

    def save_to_astore(self, filename):
        '''
        Function to save the model to an astore object, and write it into a file.
        
         Parameters:

        ----------
        filename: str
            Specify the name of the file to write.
        '''

        sess = self.sess

        if not sess.queryactionset('astore')['astore']:
            sess.loadactionset('astore')
        CAS_tbl_name = random_name('Model_astore')
        sess.dlexportmodel(casout=CAS_tbl_name,
                           initWeights=self.model_weights,
                           modelTable=self.model_name)
        model_astore = sess.download(rstore=CAS_tbl_name)
        with open(filename, 'wb') as file:
            file.write(model_astore['blob'])

    def save_to_table(self, file_name):

        '''
        Function to save the model as sas dataset.

        Parameters:

        ----------
        model_tbl_file: str
            Specify the name of the file to store the model table.
        model_tbl_file: str, optional
            Specify the name of the file to store the model weights table.
        model_tbl_file: str, optional
            Specify the name of the file to store the attribute of the model weights.


        Return:

        ----------
        The specified files in the 'CASUSER' library.

        '''

        _file_name_, _extension_ = file_name.rsplit('.', 1)
        model_tbl_file = file_name
        weight_tbl_file = _file_name_ + '_weight' + _extension_
        attr_tbl_file = _file_name_ + '_weight_attr' + _extension_

        sess = self.sess

        sess.table.save(table=self.model_name,
                        name=model_tbl_file,
                        replace=True, caslib='CASUSER')
        sess.table.save(table=self.model_weights,
                        name=weight_tbl_file,
                        replace=True, caslib='CASUSER')
        CAS_tbl_name = random_name('Attr_Tbl')
        sess.table.attribute(caslib='CASUSERHDFS', task='CONVERT', name=self.model_weights.name,
                             attrtable=CAS_tbl_name)
        sess.table.save(table=CAS_tbl_name,
                        name=attr_tbl_file,
                        replace=True, caslib='CASUSER')


class Feature_Maps:
    '''
    A class for feature maps.
    '''

    def __init__(self, sess, feature_maps_tbl, structure=None):
        self.sess = sess
        self.tbl = feature_maps_tbl
        self.structure = structure

    def display(self, layer_id):
        '''
        Function to display the feature maps.

        Parameters:

        ----------

        layer_id : int
            Specify the id of the layer to be displayed

        Return:

        ----------
        Plot of the feature maps.


        '''
        n_images = self.structure[layer_id]
        if n_images > 64:
            n_col = int(np.ceil(np.sqrt(n_images)))
        else:
            n_col = min(n_images, 8)
        n_row = int(np.ceil(n_images / n_col))

        fig = plt.figure(figsize=(16, 16 // n_col * n_row))
        title = '_LayerAct_{}'.format(layer_id)

        for i in range(n_images):
            col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, i)
            image = self.sess.fetchimages(table=self.tbl, image=col_name).Images.Image[0]
            image = np.asarray(image)
            fig.add_subplot(n_row, n_col, i + 1)
            plt.imshow(image, cmap='gray')
            plt.xticks([]), plt.yticks([])
        plt.suptitle('{}'.format(title))
