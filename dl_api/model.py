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
import os


class Model:
    '''
    Model

    Parameters:

    ----------
    conn :
        Specifies the CAS connection.
    model_name : string
        Specifies the name of the deep learning model.
    model_weights : string, dictionary or CAS table, optional
        Specifies the weights of the deep learning model.
        If not specified, random initial will be used.
        Default : None

    Returns

    -------
    A deep learning model objects.
    '''

    def __init__(self, conn, model_name=None, model_weights=None):

        if not conn.queryactionset('deepLearn')['deepLearn']:
            conn.loadactionset(actionSet='deepLearn', _messagelevel='error')

        # self.table = conn.CASTable(model_name)
        self.conn = conn
        if model_name is None:
            self.model_name = random_name('Model', 6)
        elif type(model_name) is not str:
            raise TypeError('model_name has to be a string type.')
        else:
            self.model_name = model_name

        if model_weights is None:
            self.model_weights = conn.CASTable('{}_weights'.format(self.model_name))
        else:
            self.set_weights(model_weights)

        self.valid_res = None
        self.valid_feature_maps = None
        self.valid_conf_mat = None
        self.valid_score = None
        self.n_Epoch = 0
        self.training_history = None

    def load(self, path):
        '''
        Function to load the deep learning model architecture from existing table.

        Parameters:

        ----------
        path: str
            Specifies the full path of the table.
            Note: the path need to be in Linux path format.
        '''
        conn = self.conn

        dir_name, file_name = path.rsplit('/', 1)

        CAS_lib_name = random_name('Caslib', 6)
        conn.addCaslib(_messagelevel='error',
                       name=CAS_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        conn.table.loadTable(_messagelevel='error',
                             caslib=CAS_lib_name,
                             path=file_name,
                             casout=dict(name=self.model_name,
                                         replace=True))

        model_name = conn.fetch(_messagelevel='error',
                                table=dict(name=self.model_name,
                                           where='_DLKey1_= "modeltype"')).Fetch['_DLKey0_'][0]

        if model_name.lower() != self.model_name.lower():
            conn.partition(casout=dict(name=model_name, replace=True),
                           table=self.model_name)

            conn.droptable(_messagelevel='error',
                           table=self.model_name)

            print('NOTE: Model table is loaded successfully!\n'
                  'NOTE: Model is renamed to "{}" according to the model name in the table.'.format(model_name))
            self.model_name = model_name
            self.model_weights = conn.CASTable('{}_weights'.format(self.model_name))

        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(conn.fileinfo(caslib=CAS_lib_name, includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_weights' + _extension_) in _file_name_list_:
            conn.table.loadTable(_messagelevel='error',
                                 caslib=CAS_lib_name,
                                 path=_file_name_ + '_weights' + _extension_,
                                 casout=dict(name=self.model_name + '_weights',
                                             replace=True))
            self.set_weights(self.model_name + '_weights')

            if (_file_name_ + '_weights_attr' + _extension_) in _file_name_list_:
                conn.table.loadTable(_messagelevel='error',
                                     caslib=CAS_lib_name,
                                     path=_file_name_ + '_weights_attr' + _extension_,
                                     casout=dict(name=self.model_name + '_weights_attr',
                                                 replace=True))
                self.set_weights_attr(self.model_name + '_weights_attr')

        conn.dropcaslib(_messagelevel='error', caslib=CAS_lib_name)

    def set_weights(self, weight_tbl):

        '''
        Function to assign the weight to the model.


        Parameters:

        ----------
        weight_tbl : A CAS table object, a string specifies the name of the CAS table,
                   a dictionary specifies the CAS table.
            Specifies the weights for the model.

        '''

        conn = self.conn

        weight_tbl = input_table_check(weight_tbl)
        weight_name = self.model_name + '_weights'

        if weight_tbl['name'].lower() != weight_name.lower():
            conn.partition(casout=dict(name=self.model_name + '_weights', replace=True),
                           table=weight_tbl)

        self.model_weights = conn.CASTable(name=self.model_name + '_weights')
        print('NOTE: Model weights are attached successfully!')

    def load_weights(self, path):

        '''
        Function to load the weights form a file.


        Parameters:

        ----------
        path : str
            Specifies the directory of the file that store the weight table.

        '''

        conn = self.conn

        dir_name, file_name = path.rsplit('/', 1)

        CAS_lib_name = random_name('Caslib', 6)
        conn.addCaslib(_messagelevel='error',
                       name=CAS_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        conn.table.loadTable(_messagelevel='error',
                             caslib=CAS_lib_name,
                             path=file_name,
                             casout=dict(name=self.model_name + '_weights',
                                         replace=True))

        self.set_weights(self.model_name + '_weights')

        _file_name_, _extension_ = os.path.splitext(file_name)

        _file_name_list_ = list(conn.fileinfo(caslib=CAS_lib_name, includeDirectories=False).FileInfo.Name)

        if (_file_name_ + '_attr' + _extension_) in _file_name_list_:
            conn.table.loadTable(_messagelevel='error',
                                 caslib=CAS_lib_name,
                                 path=_file_name_ + '_weights_attr' + _extension_,
                                 casout=dict(name=self.model_name + '_weights_attr',
                                             replace=True))

            self.set_weights_attr(self.model_name + '_weights_attr')

        self.model_weights = conn.CASTable(name=self.model_name + '_weights')

        conn.dropcaslib(_messagelevel='error',
                        caslib=CAS_lib_name)

    def set_weights_attr(self, attr_tbl, clear=True):

        '''
        Function to attach the weights attribute.

        Parameters:

        ----------
        attr_tbl : castable parameter
            Specifies the weights attribute table.
        clear : boolean, optional
            Specifies whether to drop the attribute table after attach it into the weight table.

        '''

        self.conn.retrieve('table.attribute', _messagelevel='error',
                           task='ADD', attrtable=attr_tbl,
                           **self.model_weights.to_table_params())

        if clear:
            self.conn.retrieve('table.droptable', _messagelevel='error',
                               table=attr_tbl)

        print('NOTE: Model attributes are attached successfully!')

    def load_weights_attr(self, path):

        '''
        Function to load the weights attribute form a file.


        Parameters:

        ----------
        path : str
            Specifies the directory of the file that store the weight attribute table.

        '''

        conn = self.conn

        dir_name, file_name = path.rsplit('/', 1)

        CAS_lib_name = random_name('Caslib', 6)
        conn.addCaslib(_messagelevel='error',
                       name=CAS_lib_name, path=dir_name, activeOnAdd=False, dataSource=dict(srcType="DNFS"))

        conn.table.loadTable(_messagelevel='error',
                             caslib=CAS_lib_name,
                             path=file_name,
                             casout=dict(name=self.model_name + '_weights_attr',
                                         replace=True))

        self.set_weights_attr(self.model_name + '_weights_attr')

        conn.dropcaslib(_messagelevel='error', caslib=CAS_lib_name)

    def model_info(self):

        '''
        Function to return the information of the model table.
        '''

        return self.conn.modelinfo(modelTable=self.model_name)

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
            Specifies the training data for the model.
        inputs : string, optional
            Specifies the variable name of in the input_tbl, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specifies the variable name of in the input_tbl, that is the response of the deep learning model.
            Default : '_label_'.
        miniBatchSize : integer, optional
            Specifies the number of observations per thread in a mini-batch..
            Default : 1.
        maxEpochs : int64, optional
            Specifies the maximum number of Epochs.
            Default : 5.
        logLevel : int 0-3, optional
            Specifies  how progress messages are sent to the client.
                0 : no messages are sent.
                1 : send the start and end messages.
                2 : send the iteration history for each Epoch.
                3 : send the iteration history for each batch.
            Default : 3.
        optimizer: dictionary, optional
            Specifies the options for the optimizer in the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_dlcommon_dlOptimizerOpts.html#type_synchronous
            for detail.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dltrain action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlTrain.html
            for detail.

        Returns

        ----------
        Retrun a fetch result to the client, about the trained model summary.
        The updated model weights are automatically assigned to the Model object.

        '''
        conn = self.conn
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
                             logLevel=logLevel)
        else:
            key_args = ['miniBatchSize', 'maxEpochs', 'logLevel']
            for key_arg in key_args:
                if optimizer[key_arg] is None:
                    optimizer[key_arg] = eval(key_arg)

        if self.model_weights.to_table_params()['name'].upper() in list(conn.tableinfo().TableInfo.Name):
            init_weights = self.model_weights
        else:
            init_weights = None
            print('NOTE: Training from scratch.')

        r = conn.dltrain(model=self.model_name,
                         table=input_tbl,
                         inputs=inputs,
                         target=target,
                         initWeights=init_weights,
                         modelWeights=dict(**self.model_weights.to_table_params(), replace=True),
                         optimizer=optimizer,
                         **kwargs)
        temp = r.OptIterHistory
        temp.Epoch += 1  # Epochs should start from 1
        temp.Epoch = temp.Epoch.astype('int64')  # Epochs should be integers

        if self.n_Epoch == 0:
            self.n_Epoch = maxEpochs
            self.training_history = temp
        else:
            temp.Epoch += self.n_Epoch
            self.training_history = self.training_history.append(temp)
            self.n_Epoch += maxEpochs
        self.training_history.index = range(0, self.n_Epoch)

        return r

    def plot_training_history(self, items=('Loss', 'FitError'), fig_size=(12, 5)):

        '''
        Function to display the training iteration history.
        '''

        self.training_history.plot(x=['Epoch'], y=list(items),
                                   xticks=self.training_history.Epoch,
                                   figsize=fig_size)

    def predict(self, data, inputs='_image_', target='_label_', **kwargs):

        '''
        Function of scoring the deep learning model on a validation data set.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                      a dictionary specifies the CAS table, or an Image object.
            Specifies the validating data for the prediction.
        inputs : string, optional
            Specifies the variable name of in the data, that is the input of the deep learning model.
            Default : '_image_'.
        target : string, optional
            Specifies the variable name of in the data, that is the response of the deep learning model.
            Default : '_label_'.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.


        '''

        conn = self.conn
        input_tbl = input_table_check(data)

        valid_res_tbl = random_name('Valid_Res')

        res = conn.dlscore(model=self.model_name, initWeights=self.model_weights,
                           table=input_tbl,
                           copyVars=[inputs, target],
                           randomFlip='NONE',
                           randomCrop='NONE',
                           casout=dict(name=valid_res_tbl, replace=True),
                           encodeName=True,
                           **kwargs)

        self.valid_score = res.ScoreInfo
        self.valid_conf_mat = conn.crosstab(
            table=valid_res_tbl, row=target, col='I__label_')

        temp_tbl = conn.CASTable(valid_res_tbl)
        temp_columns = temp_tbl.columninfo().ColumnInfo.Column

        columns = [item for item in temp_columns if item[0:9] == 'P__label_' or item == 'I__label_']
        img_table = conn.image.fetchimages(_messagelevel='error', fetchimagesvars=columns,
                                           imagetable=temp_tbl, to=1000)
        img_table = img_table.Images

        self.valid_res = img_table

        return self.valid_score

    def plot_predict_res(self, type='A', image_id=0):
        '''
        Function to plot the classification results.

        Parameters:

        ----------
        type : str, optional.
            Specifies the type of classification results to plot.
        image_id : int, optional.
            Specifies the image to be displayed.

        '''
        from .utils import plot_predict_res

        if type == 'A':
            img_table = self.valid_res
        elif type == 'C':
            img_table = self.valid_res
            img_table = img_table[img_table['Label'] == img_table['I__label_']]
        elif type == 'M':
            img_table = self.valid_res
            img_table = img_table[img_table['Label'] != img_table['I__label_']]
        else:
            raise ValueError('type must be one of the following:\n'
                             'A: for all the images\n'
                             'C: for correctly classified images\n'
                             'M: for misclassified images\n')
        img_table.index = range(len(img_table.index))

        columns_for_pred = [item for item in img_table.columns if item[0:9] == 'P__label_']

        image = img_table['Image'][image_id]
        label = img_table['Label'][image_id]

        labels = [item[9:].title() for item in columns_for_pred]
        values = np.asarray(img_table[columns_for_pred].iloc[image_id])
        values, labels = zip(*sorted(zip(values, labels)))
        values = values[-5:]
        labels = labels[-5:]
        labels = [item[:(item.find('__') > 0) * item.find('__') + (item.find('__') < 0) * len(item)]
                  for item in labels]
        labels = [item.replace('_', '\n') for item in labels]

        plot_predict_res(image, label, labels, values)

    def get_feature_maps(self, data, label=None, image_id=0, **kwargs):
        '''
        Function to extract the feature maps for a single image.

        Parameters:

        ----------
        data : An ImageTable object.
            Specifies the table containing the image data.
        label: str, optional
            Specifies the which class of image to use.
            Default : None
        image_id : int, optional
            Specifies which image to use in the table.
            Default : 1.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        Retrun an instance variable of the Model object, which is a feature map object.
        '''

        conn = self.conn
        uid = data._uid_
        if label is None:
            label = uid.iloc[0, 0]
        uid = uid.loc[uid['_label_'] == label]

        if image_id >= uid.shape[0]:
            raise ValueError('image_id should be an integer between 0'
                             ' and {} here.'.format(uid.shape[0] - 1))
        uid_value = uid.iloc[image_id, 1]
        uid_name = uid.columns[1]

        input_tbl = input_table_check(data)

        feature_maps_tbl = random_name('Feature_Maps') + '_{}'.format(image_id)
        conn.dlscore(model=self.model_name, initWeights=self.model_weights,
                     table=dict(**input_tbl,
                                where='{}="{}"'.format(uid_name, uid_value)),
                     layerOut=dict(name=feature_maps_tbl, replace=True),
                     randomFlip='NONE',
                     randomCrop='NONE',
                     layerImageType='jpg',
                     encodeName=True,
                     **kwargs)
        layer_out_jpg = conn.CASTable(feature_maps_tbl)
        feature_maps_names = [i for i in layer_out_jpg.columninfo().ColumnInfo.Column]
        feature_maps_structure = dict()
        for feature_map_name in feature_maps_names:
            feature_maps_structure[int(feature_map_name.split('_')[2])] = int(feature_map_name.split('_')[4]) + 1

        self.valid_feature_maps = Feature_Maps(self.conn, feature_maps_tbl, structure=feature_maps_structure)

    def get_features(self, data, dense_layer, target='_label_', **kwargs):
        '''
        Function to extract the features for a data table.

        Parameters:

        ----------
        data : A CAS table object, a string specifies the name of the CAS table,
                    a dictionary specifies the CAS table, or an Image object.
            Specifies the table containing the image data.
        dense_layer : str
            Specifies the name of the layer that is extracted.
        target : str, optional
            Specifies the name of the column including the response variable.
        kwargs: dictionary, optional
            Specifies the optional arguments for the dlScore action.
            see http://casjml01.unx.sas.com:8080/job/Actions_ref_doc_latest/ws/casaref/casaref_python_tkcasact_deepLearn_dlScore.html
            for detail.

        Returns

        ----------
        x : ndarray of size n by p, where n is the sample size and p is the number of features.
            The features extracted by the model at the specified dense_layer.
        y : ndarray of size n.
            The response variable of the original data.
        '''

        conn = self.conn
        input_tbl = input_table_check(data)
        feature_tbl = random_name('Features')

        conn.dlscore(model=self.model_name, initWeights=self.model_weights,
                     table=dict(**input_tbl),
                     layerOut=dict(name=feature_tbl, replace=True),
                     layerList=dense_layer,
                     layerImageType='wide',
                     randomFlip='NONE',
                     randomCrop='NONE',
                     encodeName=True,
                     **kwargs)
        x = conn.CASTable(feature_tbl).as_matrix()
        y = conn.CASTable(**input_tbl)[target].as_matrix().ravel()

        return x, y

    def save_to_astore(self, path):
        '''
        Function to save the model to an astore object, and write it into a file.

         Parameters:

        ----------
        path: str
            Specifies the name of the path to store the model astore.
        '''
        conn = self.conn

        if not conn.queryactionset('astore')['astore']:
            conn.loadactionset('astore', _messagelevel='error')

        CAS_tbl_name = random_name('Model_astore')

        conn.dlexportmodel(_messagelevel='error',
                           casout=CAS_tbl_name,
                           initWeights=self.model_weights,
                           modelTable=self.model_name)

        model_astore = conn.download(_messagelevel='error',
                                     rstore=CAS_tbl_name)

        file_name = self.model_name + '.astore'
        path = os.path.abspath(path)

        if not os.path.isdir(path):
            os.makedirs(path)
        file_name = os.path.join(path, file_name)
        with open(file_name, 'wb') as file:
            file.write(model_astore['blob'])

    def save_to_table(self, path):

        '''
        Function to save the model as sas dataset.

        Parameters:

        ----------
        path: str
            Specifies the name of the path to store the model tables.

        Return:

        ----------
        The specified files in the 'CASUSER' library.

        '''

        conn = self.conn

        CAS_lib_name = random_name('CASLIB')
        conn.addcaslib(_messagelevel='error',
                       activeonadd=False, datasource=dict(srcType="DNFS"), name=CAS_lib_name,
                       path=path)

        _file_name_ = self.model_name.replace(' ', '_')
        _extension_ = '.sashdat'
        model_tbl_file = _file_name_ + _extension_
        weight_tbl_file = _file_name_ + '_weights' + _extension_
        attr_tbl_file = _file_name_ + '_weights_attr' + _extension_

        conn = self.conn

        conn.table.save(_messagelevel='error',
                        table=self.model_name,
                        name=model_tbl_file,
                        replace=True, caslib=CAS_lib_name)
        conn.table.save(_messagelevel='error',
                        table=self.model_weights,
                        name=weight_tbl_file,
                        replace=True, caslib=CAS_lib_name)
        CAS_tbl_name = random_name('Attr_Tbl')
        conn.table.attribute(_messagelevel='error',
                             task='CONVERT', **self.model_weights.to_table_params(),
                             attrtable=CAS_tbl_name)
        conn.table.save(_messagelevel='error',
                        table=CAS_tbl_name,
                        name=attr_tbl_file,
                        replace=True, caslib=CAS_lib_name)

        conn.dropcaslib(_messagelevel='error',
                        caslib=CAS_lib_name)

    def deploy(self, path, output_format='ASTORE'):
        '''
        Function to deploy the deep learning model.

        Parameters:

        ----------
        path : string,
            Specifies the name of the path to store the model tables or astore.
        format : string, optional.
            specifies the format of the deployed model.
            Supported format: ASTORE, CASTABLE
            Default: ASTORE


        '''

        if output_format.lower() == 'astore':
            self.save_to_astore(path=path)
        elif output_format.lower() in ('castable', 'table'):
            self.save_to_table(path=path)
        else:
            raise TypeError('output_format must be "astore", "castable" or "table"')


class Feature_Maps:
    '''
    A class for feature maps.
    '''

    def __init__(self, conn, feature_maps_tbl, structure=None):
        self.conn = conn
        self.tbl = feature_maps_tbl
        self.structure = structure

    def display(self, layer_id, filter_id=None):
        '''
        Function to display the feature maps.

        Parameters:

        ----------

        layer_id : int
            Specifies the id of the layer to be displayed

        Return:

        ----------
        Plot of the feature maps.


        '''
        from PIL import Image
        from IPython.display import display

        if filter_id is None:
            n_images = self.structure[layer_id]
            filter_id = list(range(n_images))

        if len(filter_id) > 64:
            filter_id = filter_id[0:64]
            print('NOTE: The maximum number of filters to be displayed is 64.\n'
                  'NOTE: Only the first 64 filters are displayed.')

        n_images = len(filter_id)
        n_col = min(n_images, 8)
        n_row = int(np.ceil(n_images / n_col))

        fig = plt.figure(figsize=(16, 16 // n_col * n_row))
        title = 'Activation Maps for Layer_{}'.format(layer_id)

        if layer_id == 0:
            image = []
            for i in range(3):
                col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, i)
                temp = self.conn.fetchimages(_messagelevel='error',
                                             table=self.tbl, image=col_name).Images.Image[0]
                image.append(np.asarray(temp))
            image = np.dstack((image[2], image[1], image[0]))
            image = Image.fromarray(image, 'RGB')
            display(image)
        else:
            for i in range(n_images):
                filter_num = filter_id[i]
                col_name = '_LayerAct_{}_IMG_{}_'.format(layer_id, filter_num)
                image = self.conn.fetchimages(_messagelevel='error',
                                              table=self.tbl, image=col_name).Images.Image[0]
                image = np.asarray(image)
                fig.add_subplot(n_row, n_col, i + 1)
                plt.imshow(image, cmap='gray')
                plt.xticks([]), plt.yticks([])
                plt.title('Filter {}'.format(filter_num))
            plt.suptitle(title, fontsize=20)
