#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
import yaml
import inspect
import swat.datamsghandlers as dmh
from dlpy.utils import *

model_name_map = {
    'LENET': ['SAS_TORCH_LENET'],
    'EASENET': ['SAS_TORCH_EASENET'],
    'YOLOV5': ['SAS_TORCH_YOLO_V5', 'SMALL', 'MEDIUM', 'LARGE', 'EXTRALARGE'],
    'ENET': ['SAS_TORCH_ENET'],
    'DRNN': ['SAS_TORCH_DRNN'],
    'UNET': ['SAS_TORCH_UNET'],
    'RESNET': ['SAS_TORCH_RESNET', 'RESNET18', 'RESNET34', 'RESNET50', 'RESNET101'],
    'VGG': ['SAS_TORCH_VGG', 'VGG11', 'VGG13', 'VGG16', 'VGG19'],
    'MOBILENET': ['SAS_TORCH_MOBILENET'],
    'SHUFFLENET': ['SAS_TORCH_SHUFFLENET', 'SHUFFLENETX0_5', 'SHUFFLENETX1_0', 'SHUFFLENETX1_5', 'SHUFFLENETX2_0']
}

obj_detect_model_list = {'SAS_TORCH_YOLO_V5'}
rnn_model_list = {'SAS_TORCH_DRNN'}

dataset_type_list = ['UNIVARIATE', 'OBJDETECT', 'SEGMENTATION', 'AUTOENCODER']

log_level_map = {
    0: 'ERROR',
    1: 'WARNING',
    2: 'INFO',
    3: 'NOTE',
    4: 'DEBUG',
    5: 'TRACE'
}

optimizer_map = {'LearningRate': 'lr',
                 'Momentum': 'momentum',
                 'WeightDecay': 'weight_decay',
                 'Dampening': 'dampening',
                 'BETA1': 'beta1',
                 'BETA2': 'beta2'
                 }

class SGDSolver(DLPyDict):
    '''
    SGDSolver object

    Parameters
    ----------
    lr : double or HyperRange, optional
        Specifies the learning rate for the deep learning algorithm.
    dampening : double or HyperRange, optional
        Specifies the dampening for momentum of SGD method.
    momentum : double or HyperRange, optional
        Specifies the momentum factor of SGD method.
    nesterov : bool, optional
        Specifies to enable Nesterov momentum or not.
    weight_decay : double or HyperRange, optional
        Specifies the weight decay (L2 penalty) of the optimization method.

    Returns
    -------
    :class:`SGDSolver`
    '''

    def __init__(self, lr=1e-4, dampening=0, momentum=0.9, nesterov=False, weight_decay=0):
        DLPyDict.__init__(self, lr=lr, dampening=dampening, momentum=momentum,
                          nesterov=nesterov, weight_decay=weight_decay)
        self.__setitem__('method', 'sgd')

class AdamSolver(DLPyDict):
    '''
    AdamSolver object

    Parameters
    ----------
    lr : double or HyperRange, optional
        Specifies the learning rate for the deep learning algorithm.
    beta_1 : double or HyperRange, optional
        Specifies the exponential decay rate for the first moment in an ADAM learning algorithm.
    beta_2 : double or HyperRange, optional
        Specifies the exponential decay rate for the second moment in an ADAM learning algorithm.
    eps : double, optional
        Specifies the term added to the denominator to improve numerical stability.S
    amsgrad : bool, optional
        When set to True, the AMSGrad variant of the ADAM algorithm is used.
    weight_decay : double or HyperRange, optional
        Specifies the weight decay (L2 penalty) of the optimization method.

    Returns
    -------
    :class:`AdamSolver`
    '''

    def __init__(self, lr=1e-4, beta_1=0.9, beta_2=0.999, eps=1e-8, amsgrad=False, weight_decay=0):
        DLPyDict.__init__(self, lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps, amsgrad=amsgrad,
                          weight_decay=weight_decay)
        self.__setitem__('method', 'adam')

class AdamWSolver(DLPyDict):
    '''
    AdamWSolver object

    Parameters
    ----------
    lr : double or HyperRange, optional
        Specifies the learning rate for the deep learning algorithm.
    beta_1 : double or HyperRange, optional
        Specifies the exponential decay rate for the first moment in an ADAM learning algorithm.
    beta_2 : double or HyperRange, optional
        Specifies the exponential decay rate for the second moment in an ADAM learning algorithm.
    eps : double, optional
        Specifies the term added to the denominator to improve numerical stability.S
    amsgrad : bool, optional
        When set to True, the AMSGrad variant of the ADAM algorithm is used.
    weight_decay : double or HyperRange, optional
        Specifies the weight decay (L2 penalty) of the optimization method.

    Returns
    -------
    :class:`AdamWSolver`
    '''

    def __init__(self, lr=1e-4, beta_1=0.9, beta_2=0.999, eps=1e-8, amsgrad=False, weight_decay=0):
        DLPyDict.__init__(self, lr=lr, beta_1=beta_1, beta_2=beta_2, eps=eps, amsgrad=amsgrad,
                          weight_decay=weight_decay)
        self.__setitem__('method', 'adamw')

class AdagradSolver(DLPyDict):
    '''
    AdagradSolver object

    Parameters
    ----------
    lr : double or HyperRange, optional
        Specifies the learning rate for the deep learning algorithm.
    eps : double, optional
        Specifies the term added to the denominator to improve numerical stability.
    lr_decay : double, optional
        Specifies the learning rate decay of the optimization method.

    Returns
    -------
    :class:`AdagradSolver`
    '''

    def __init__(self, lr=1e-4, eps=1e-8, weight_decay=1, lr_decay=0):
        DLPyDict.__init__(self, lr=lr, eps=eps, weight_decay=weight_decay, lr_decay=lr_decay)
        self.__setitem__('method', 'adagrad')

class Mode(DLPyDict):
    '''
    Mode object

    Parameters
    ----------
    type : string, optional
    sync_freq : int, optional

    Returns
    -------
    :class:`Mode`
    '''

    def __init__(self, type='synchronous', sync_freq=1):
        DLPyDict.__init__(self, type=type, syncFreq=sync_freq)

class Optimizer(DLPyDict):
    '''
    Optimizer object

    Parameters
    ----------
    algorithm : Algorithm, optional
        Specifies the deep learning algorithm.
    loss_func : string, optional
        Specifies the loss function for the optimization method.
        Possible values: ['cross_entropy', 'mse', 'nll']
    batch_size : int or BatchSizeRange, optional
        Specifies the number of observations in a batch.
    max_epochs : int, optional
        Specifies the maximum number of epochs.
    seed : int, optional
        specifies the random number seed for the random number generator.
        The default value, 0, and negative values indicate to use random number streams based on the computer clock.
        Specify a value that is greater than 0 for a reproducible random number sequence.

    Returns
    -------
    :class:`Optimizer`
    '''

    def __init__(self, algorithm=SGDSolver(), mode=Mode(), loss_func='cross_entropy', batch_size=1, max_epochs=5, seed=0):
        DLPyDict.__init__(self, algorithm=algorithm, mode=mode, lossFunc=loss_func, batchSize=batch_size, maxEpochs=max_epochs,
                          seed=seed)

class LRScheduler(DLPyDict):
    '''
    LRScheduler object

    Parameters
    ----------
    policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.

    Returns
    -------
    :class:`LRScheduler`
    '''

    def __init__(self, policy='STEP', step_size=10, gamma=0.1):
        DLPyDict.__init__(self, policy=policy, stepSize=step_size, gamma=gamma)

class Tuner(DLPyDict):
    '''
    Tuner object

    Parameters
    ----------
    method : string, optional
        Specifies the hyperparameter tuning search method.
        Valid Values: LHS, EAGLS, RANDOM, GRID
    max_func : int, optional
        Specifies the maximum func.
    max_iterations : int, optional
        Specifies the maximum interations.
    max_time : int, optional
        Specifies the maximum time.
    pop_size : int, optional
        Specifies the population size.
    seed, int, optional
        Specifies the random number seed for the random number generator.
    Returns
    -------
    :class:`Tuner`
    '''

    def __init__(self, method='RANDOM', max_func=0, max_iterations=10, max_time=0, pop_size=20, seed=0):
        DLPyDict.__init__(self, searchMethod=method, maxFunc=max_func, maxIterations=max_iterations, maxTime=max_time,
                          popSize=pop_size, seed=seed)

class HyperRange(DLPyDict):
    '''
    HyperRange object

    Parameters
    ----------
    lower : double, optional
        Specifies the lower bound of the hyperparameter.
    upper : double, optional
        Specifies the upper bound of the hyperparameter.
    log_scale : string, optional
        Specifies the log scale of the hyperparatemer.
        Valid Values: NONE, LINEAR, LOG10, LN, FACTOR
    scale_factor : double, optoinal
        Specifies the scale factor of the hyperparatemer.
    value_list : array, optional
        Specifies the value list of the hyperparatemer.
    Returns
    -------
    :class:`HyperRange`
    '''

    def __init__(self, lower=None, upper=None, log_scale=None, scale_factor=None, value_list=None):
        DLPyDict.__init__(self, lb=lower, ub=upper, logScale=log_scale, scaleFactor=scale_factor, valueList=value_list)

class BatchSizeRange(DLPyDict):
    '''
    BatchSizeRange object

    Parameters
    ----------
    lower : int, optional
        Specifies the lower bound of the batch size.
    upper : int, optional
        Specifies the upper bound of the batch size
    value_list : array, optional
        Specifies the value list of the batch size.
    Returns
    -------
    :class:`BatchSizeRange`
    '''

    def __init__(self, lower=None, upper=None, value_list=None):
        DLPyDict.__init__(self, lb=lower, ub=upper, valueList=value_list)

class MZModelCal(dmh.CASDataMsgHandler):
    """
    Parameters
    ----------
    model_local_path : string
        The path to the model path on client side.
    """
    def __init__(self, model_local_path):
        self.data = [(open(model_local_path, 'rb').read(),)]
        variables = [dict(name='Data', type='VARBINARY')]  # data column
        self.model_local_path = model_local_path
        super(MZModelCal, self).__init__(variables)

    def getrow(self, row):
        try:
            return self.data[row]
        except Exception as e:
            raise DLPyError(e.message)

class MZModel():
    """
    Parameters
    ----------
    model_type : string
        Either "TorchNative" or "TorchScript".
    model_name : string, optional
        The name of the model. This filed is only used for torchnative models.
    model_subtype : string, optional
        The subtype of the model. This filed is only used for torchnative models.
    anchors : string, optional
        The initial guess of object sizes and locations.
    rnn_type : string, optional
        Specifies the type of the recurrent layer to use.
    input_size : int, optional
        Specifies the size of the input for the model.
    hidden_size : int, optional
        Specifies the number of features of the hidden state of the model.
    num_layers : int, optional
        Specifies the number of layers in the model.
    dataset_type : string
        The type of dataset used in the model.
    caslib : string, optional
        The CAS library where the model file resides.
    model_path : string, optional
        The path to the model. It could either be the cas-server side absolute path to the model, or the relative path to the caslib.
    num_classes : int, optional
        The number of classes in the dataset.
    encoding : int, optional
        Specifies the vector size for one-hot encoding.

    Returns
    --------
    :class:`CASResults`
    """

    def __init__(self, conn, model_type, model_name=None, model_subtype=None, anchors=None, rnn_type=None,
                 input_size=None, hidden_size=None, num_layers=None, dataset_type="Univariate", caslib=None,
                 model_path=None, num_classes=10, encoding=0):
        self.conn = conn
        self.model_table = None
        self._init_model(model_type, model_name, model_subtype, anchors, rnn_type, input_size, hidden_size, num_layers,
                         dataset_type, caslib, model_path, num_classes, encoding)
        self.conn.loadactionset(actionSet='dlModelZoo', _messagelevel='error')

    def _init_model(self, model_type, model_name, model_subtype, anchors, rnn_type, input_size, hidden_size, num_layers,
                    dataset_type, caslib, model_path, num_classes, encoding):
        if model_type.upper() not in ['TORCHNATIVE', 'TORCHSCRIPT']:
            raise DLPyError('Model type should be either TorchNative or TorchScript.')

        model_type = model_type.upper()
        if model_type == "TORCHNATIVE":
            if not model_name:
                raise DLPyError('Model name is required for TorchNative models.')
            model_name = model_name.upper()
            if model_name not in model_name_map:
                raise DLPyError('TorchNative model ' + model_name + ' does not exit. Please check your model name or use Torchscript model.')
            if model_subtype is not None:
                model_subtype = model_subtype.upper()
                if model_subtype is not None and len(model_name_map[model_name]) > 1:
                    if model_subtype not in model_name_map[model_name][1:]:
                        raise DLPyError('Subtype ' + model_subtype + ' does not exit. Please check your model subtype.')

        dataset_type = dataset_type.upper()
        if dataset_type not in dataset_type_list:
            raise DLPyError('Dataset type ' + dataset_type + ' does not exit. Please check your dataset type.')

        if model_path is not None:
            if os.path.isabs(model_path):
                with caslibify_context(self.conn, model_path, task = 'load') as (caslib_created, path_created):
                    caslib = caslib_created
                    model_path = path_created

                    if caslib is None and model_path is None:
                        print('Cannot create a caslib for the provided path. Please make sure that the path is accessible from'
                              'the CAS Server. Please also check if there is a subpath that is part of an existing caslib')
            elif caslib is None:
                raise DLPyError('Please specify caslib if using relative path.')

        self.model_table_name = random_name('Model', 6)
        self.index_map_name = random_name('Table', 6)

        self.label_name = random_name('Doc', 6)

        train_filename = os.path.join('datasources', 'mztrain.yaml')
        score_filename = os.path.join('datasources', 'mzscore.yaml')
        project_path = os.path.dirname(os.path.abspath(__file__))
        train_yaml_dir = os.path.join(project_path, train_filename)
        score_yaml_dir = os.path.join(project_path, score_filename)

        with open(train_yaml_dir) as file:
            self.documents_train = yaml.full_load(file)
        with open(score_yaml_dir) as file:
            self.documents_score = yaml.full_load(file)

        self.documents_train['sas']['dlx']['train']['label'] = self.label_name + "_train"
        self.documents_train['sas']['dlx']['train']['dataset']['type'] = dataset_type

        self.documents_score['sas']['dlx']['score']['label'] = self.label_name + "_score"
        self.documents_score['sas']['dlx']['score']['dataset']['type'] = dataset_type

        model = {}
        model['type'] = model_type
        if model_type == "TORCHNATIVE":
            model['name'] = model_name_map[model_name][0]
            model['subtype'] = model_subtype
        model['classNumber'] = num_classes
        model['anchors'] = anchors
        model['rnnType'] = rnn_type
        model['inputSize'] = input_size
        model['hiddenSize'] = hidden_size
        model['numLayers'] = num_layers

        if model_path is not None:
            model['caslib'] = caslib
            model['path'] = model_path

        for (key, value) in model.items():
            self.documents_train['sas']['dlx']['train']['model'][key] = value
            self.documents_score['sas']['dlx']['score']['model'][key] = value

        if encoding > 0:
            self.documents_train['sas']['dlx']['train']['model']['outputs'][0]['size'] = [encoding]
            self.documents_score['sas']['dlx']['score']['model']['outputs'][0]['size'] = [encoding]

    def add_image_transformation(self, image_resize_type='To_FIX_DIM', image_size=None, target_size=None,
                                 color_transform=None, random_transform=False):
        """
        Add image transformation to yaml file

        Parameters
        ----------
        image_resize_type : string, optional
            Specifies the type to resize images.
        image_size : string, optional
            Specifies the size of the input images.
        target_size : string, optional
            Specifies the size of the target images. It is only used for segmentaiton dataset.
        color_transform : string, optional
            Specifies the type of color space transformation to be performed for input images.
        random_transform : bool, optional
            Specifies whether to apply random transform to input images.
        """

        image_dict = {}
        if image_size is not None:
            image_dict['resize'] = {'type': image_resize_type, 'size': image_size}
            if target_size is not None:
                image_dict['resize']['target_size'] = target_size
        if color_transform is not None:
            image_dict['colorTransform'] = color_transform
        if random_transform is True:
            image_dict['randomTransform']['mosaicProb'] = 0.2
            image_dict['randomTransform']['random_flip_h'] = 0.01
            image_dict['randomTransform']['random_flip_v'] = 0.01
            image_dict['randomTransform']['degree'] = 0.373
            image_dict['randomTransform']['translate'] = 0.245
            image_dict['randomTransform']['scale'] = 0.2
            image_dict['randomTransform']['shear'] = 0.302
            image_dict['randomTransform']['perspective'] = 0.003
            image_dict['randomTransform']['colorSpace'] = "0.1 0.12 0.15"


        if self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['imageTransformation'] is None:
            self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['imageTransformation'] = {}
            self.documents_score['sas']['dlx']['score']['preProcessing'][0]['modelInput']['imageTransformation'] = {}

        for (key, value) in image_dict.items():
            self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['imageTransformation'][key] = value
            self.documents_score['sas']['dlx']['score']['preProcessing'][0]['modelInput']['imageTransformation'][key] = value

    def add_text_transformation(self, word_embedding="word2Vec"):
        """
        Add image transformation to yaml file

        Parameters
        ----------
        word_embedding : string, optional
            Specifies word embedding operation
        """
        text_dict = {}
        text_dict['word_embedding'] = word_embedding

        if self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['textTransformation'] is None:
            self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['textTransformation'] = {}
            self.documents_score['sas']['dlx']['score']['preProcessing'][0]['modelInput']['textTransformation'] = {}

        for (key, value) in text_dict.items():
            self.documents_train['sas']['dlx']['train']['preProcessing'][0]['modelInput']['textTransformation'][key] = value
            self.documents_score['sas']['dlx']['score']['preProcessing'][0]['modelInput']['textTransformation'][key] = value

    def init_index(self, index_variable, index_map):
        if 'index_variable' not in dir(self):
            self.index_variable = None
        if 'index_map' not in dir(self):
            self.index_map = None

        if index_variable is not None:
            self.index_variable = index_variable
        if index_map is not None:
            self.index_map = index_map

        if self.index_map is not None and self.index_variable is None:
            raise DLPyError('Please specify index variable if using an index map.')

    def train(self, table, model=None, inputs=None, targets=None, index_variable=None, batch_size=1,
              max_epochs=5, log_level=0, lr=0.01, optimizer=None, valid_table=None, gpu=None, seed=0, n_threads=None,
              drop_last=False, lr_scheduler=None, tuner=None):
        """
        Train a deep learning model.

        Note that this function surfaces several parameters from other parameters. For example,
        while learning rate is a parameter of Optimizer), it is leveled up so that our users can easily set learning
        rate without changing the default optimizer and solver. If a non-default solver or optimizer is passed,
        then these leveled-up parameters will be ignored - even they are set - and the ones coming from
        the custom solver and custom optimizer will be used. In addition to learning_rate(lr),
        max_epochs and seed are another examples of such parameters.

        Parameters
        ----------
        table : string or CASTable, optional
            This is the input data.
        model: string, optinal
            The CAS table containing the model and model weights.
            When it is specified, the table stores a binary blob containing the model and weights in Pytorch format.
        inputs : string
            The input variables for the training task.
            The input column can either be a string of a image path or a binary blob of an image data.
        targets : string
            The target variables for the training task.
            The targets can either specify a single column of path string, which points to txt files containing targets,
            or a list of columns containing the target values, e.g, class names and object locations, etc.
        index_variable : string
            The variables to convert to numeric indexes.
        batch_size : int, optional
            Specifies the number of observations in a batch.
        max_epochs : int, optional
            Specifies the maximum number of epochs.
        log_level : int, optional
            The reporting level for progress messages sent to the client.
            The default level 0(ERROR) indicates that only error messages are sent.
            Setting the value to 1(WARNING) sends start, end, and warning messages.
            Setting the value to 2(NOTE) adds the iteration history to the client messaging;
            a summary of the model used is also printed on the journal at the 'NOTE' level.
            Setting the value to 3(INFO) adds batch-level information to the client messaging.
            Setting the value to 4(DEBUG), more information related to the training process is printed.
            Setting the value to 5(TRACE), a detailed model architecture is printed on the journal.
        lr : double, optional
            Specifies the learning rate.
        optimizer : class:`Optimizer`, optional
            Specifies the parameters for the optimizer.
        valid_table : string or CASTable, optional
            Specifies the table with the validation data.
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing unit hardware.
        seed : int, optional
            Specifies the random number seed for the random number generator.
            The default value, 0, and negative values indicate to use random number streams based on the computer clock.
            Specify a value that is greater than 0 for a reproducible random number sequence.
        n_threads : int, optional
            Specifies the number of threads to use.
        drop_last : bool, optional
            Specifies whether to skip the last batch if it is not a full batch.
        lr_scheduler : class:`LRScheduler`, optional
            Specifies the learning rate policy.
        tuner : class:`Tuner`, optional
            Specifies the tuning method.

        Returns
        --------
        :class:`CASResults`
        """

        if optimizer is None:
            if 'optimizer' in dir(self):
                optimizer = self.optimizer
            else:
                optimizer = Optimizer(algorithm=SGDSolver(lr=lr),  batch_size=batch_size, max_epochs=max_epochs, seed=seed)

        self.inputs = inputs
        
        if model is None and self.model_table:
            model = self.model_table

        parameters = DLPyDict(logLevel=log_level_map[log_level], table=table, inputs=inputs, targets=targets,
                              valid_table=valid_table, indexvariables = index_variable,
                              outputIndexmap=dict(name=self.index_map_name, replace=True), model=model, gpu=gpu, optimizer=optimizer,
                              n_threads=n_threads,
                              options=dict(yaml=str(self.documents_train), label=self.label_name + "_train"),
                              modelOut=dict(name=self.model_table_name, replace=True), dropLast=drop_last, learningRateScheduler=lr_scheduler,
                              tuner=tuner)

        rt = self.conn.retrieve('dlModelZoo.dlmztrain', _messagelevel='note', **parameters)

        # update model table to the trained model
        self.model_table = self.conn.CASTable(self.model_table_name)
        self.index_variable = None
        self.index_map = None
        if index_variable is not None:
            self.index_variable = index_variable
            self.index_map = self.conn.CASTable(self.index_map_name)

        hyper_tuning = 'HyperparameterValues'
        if hyper_tuning in rt:
            self.optimizer = optimizer
            tuning_results = rt[hyper_tuning]
            tuning_list = tuning_results.columns.tolist()
            if 'BatchSize' in tuning_list:
                self.optimizer['batch_size'] = tuning_results['BatchSize'][0]
            for item in tuning_list:
                if item in optimizer_map:
                    self.optimizer['algorithm'][optimizer_map[item]] = tuning_results[item][0]

        return rt

    def score(self, table, model=None, inputs=None, targets=None, index_variable=None, index_map=None, log_level=0,
              gpu=None, n_threads=None, batch_size=None):
        """
        Score a deep learning model.

        Parameters
        ----------
        table : string
            This is the input data.
        model: string, optinal
            The CAS table containing the model and model weights.
            When it is specified, the table stores a binary blob containing the model and weights in Pytorch format.
        inputs : string
            The input variables for the training task.
            The input column can either be a string of a image path or a binary blob of an image data.
        targets : string
            The target variables for the training task.
            The targets can either specify a single column of path string, which points to txt files containing targets,
            or a list of columns containing the target values, e.g, class names and object locations, etc.
        index_variable : string
            The variables to convert to numeric indexes.
        index_map : string
            The table contains index map.
        batch_size : int, optional
            Specifies the number of observations in a batch.
        log_level : int, optional
            The reporting level for progress messages sent to the client.
            The default level 0(ERROR) indicates that only error messages are sent.
            Setting the value to 1(WARNING) sends start, end, and warning messages.
            Setting the value to 2(NOTE) adds the iteration history to the client messaging;
            a summary of the model used is also printed on the journal at the 'NOTE' level.
            Setting the value to 3(INFO) adds batch-level information to the client messaging.
            Setting the value to 4(DEBUG), more information related to the training process is printed.
            Setting the value to 5(TRACE), a detailed model architecture is printed on the journal.
        gpu : :class:`Gpu`, optional
            When specified, the action uses graphical processing unit hardware.
        n_threads : int, optional
            Specifies the number of threads to use.

        Returns
        --------
        :class:`CASResults`
        """

        if model is None and self.model_table:
            model = self.model_table

        self.init_index(index_variable, index_map)

        input_table = self.conn.CASTable(table)
        copy_vars = input_table.columns.tolist()

        temp_table_out = random_name('Table', 6)

        parameters = DLPyDict(logLevel=log_level_map[log_level], table=table, inputs=inputs, targets=targets,
                              model=model, gpu=gpu, n_threads=n_threads, batch_size=batch_size,
                              indexvariables=self.index_variable, inputIndexmap=self.index_map,
                              options=dict(yaml=str(self.documents_score), label=self.label_name + "_score"),
                              tableOut=temp_table_out, copyvars=copy_vars)

        rt = self.conn.retrieve('dlModelZoo.dlmzscore', _messagelevel='note', **parameters)

        self.table_out = self.conn.CASTable(temp_table_out)

        return rt

    def save_to_astore(self, path, file_name=None, index_variable=None, index_map=None):
        """
        Save the model to an astore object, and write it into a file.

        Parameters
        ----------
        path : string
            Specifies the client-side path to store the model astore.
            The path format should be consistent with the system of the client.
        file_name : string
            Specifies the name of the saved astore file.
        index_variable : string
            The variables to convert to numeric indexes.
        index_map : string
            The table contains index map.
        """

        self.conn.loadactionset('astore', _messagelevel='error')

        CAS_tbl_name = self.model_table_name + '_astore'

        self.init_index(index_variable, index_map)

        parameters = DLPyDict(model=self.model_table,
                        outputAstore=dict(replace=True, name=CAS_tbl_name),
                        inputs=self.inputs, indexvariables=self.index_variable, inputIndexmap=self.index_map,
                        options=dict(yaml=str(self.documents_score), label=self.label_name + "_score"))

        self.conn.retrieve('dlModelZoo.dlmzExport', _messagelevel='note', **parameters)

        model_astore = self.conn.retrieve('astore.download', rstore=CAS_tbl_name)

        if file_name is None:
            file_name = self.model_table_name + '.astore'
        else:
            file_name = file_name + '.astore'

        if path is None:
            path = os.getcwd()

        if not os.path.exists(path):
            raise DLPyError('There seems to be an error while writing the astore file to the client. '
                            'Please check out the path provided. There is also a chance that you are passing '
                            'a server side path while the function expects a client side path')
        else:
            file_name = os.path.join(path, file_name)
            file = open(file_name, 'wb')
            file.write(model_astore['blob'])
            file.close()
            print('NOTE: Model astore file saved successfully.')
            if self.index_map is not None:
                print('NOTE: Model index map saved successfully.')

    def save_to_table(self, path, file_name=None, index_map=None):
        """
        Save the model as SAS dataset

        Parameters
        ----------
        path : string
            Specifies the server-side path to store the model tables.
        file_name : string
            Specifies the name of the saved CAS table file.
        index_map : string
            The table contains index map.
        """

        self.init_index(None, index_map)

        with caslibify_context(self.conn, path, task='save') as (caslib, path_remaining):
            if file_name is None:
                _file_name_ = self.model_table_name.replace(' ', '_')
            else:
                _file_name_ = file_name.replace(' ', '_')
            _extension_ = '.sashdat'
            model_tbl_file = path_remaining + _file_name_ + _extension_
            if self.index_map is not None:
                model_index_file = path_remaining + _file_name_ + "_index" + _extension_

            if self.model_table is not None:
                rt = self.conn.retrieve('table.save', table=self.model_table, name=model_tbl_file, replace=True,
                                     caslib=caslib)
                if rt.severity > 1:
                    for msg in rt.messages:
                        print(msg)
                    raise DLPyError('something is wrong while saving the model to a table!')
                if self.index_map is not None:
                    rt = self.conn.retrieve('table.save', table=self.index_map, name=model_index_file, replace=True,
                                            caslib=caslib)
                    if rt.severity > 1:
                        for msg in rt.messages:
                            print(msg)
                        raise DLPyError('something is wrong while saving the index map to a table!')
                    print('NOTE: Model index map saved successfully.')

            print('NOTE: Model table saved successfully.')

    def load_from_table(self, path, index_path=None):
        '''
        Load the weights from a data file specified by ‘path’

        Parameters
        ----------
        path : string
            Specifies the server-side directory of the file that contains the weight table.
        index_path : string
            Specifies the server-side directory of the file that contains the index map table.

        Notes
        -----
        Currently support sashdat files.

        '''
        if not file_exist_on_server(self.conn, path):
            raise DLPyError('The file, {}, doesn\'t exist on the server-side.'.format(path))

        with caslibify_context(self.conn, path, task='load') as (cas_lib_name, file_name):
            self.conn.retrieve('table.loadtable',
                            caslib=cas_lib_name,
                            path=file_name,
                            casout=dict(replace=True, name=self.model_table_name))

            self.model_table = self.conn.CASTable(self.model_table_name)

        if index_path is not None:
            with caslibify_context(self.conn, index_path, task='load') as (cas_lib_name, file_name):
                self.conn.retrieve('table.loadtable',
                                caslib=cas_lib_name,
                                path=file_name,
                                casout=dict(replace=True, name=self.index_map_name))

                self.index_map = self.conn.CASTable(self.index_map_name)

    def upload_model_from_client(self, model_local_path: str, replace: bool =False):
        '''
        Upload a model stored on client side to CAS server. 
        It could be helpful in situations where accessing the server filesystem is either difficult or not possible.

        Parameters
        ----------
        model_local_path : string
            Specifies the client-side directory of the file that contains the weight table. 
            It could be a Torchscript file or Pytorch pickle file.
        replace : bool
            Whether replace the model table if exists.
        '''
        status = self.conn.tableExists(self.model_table_name)
        if status.exists:
            if not replace:
                print('WARNING: Model table has already existed. Set replace=True to replace the existing one.')
                return
            else:
                self.conn.droptable(self.model_table_name)
        mzweights = MZModelCal(model_local_path)
        self.conn.addtable(table=self.model_table_name, **mzweights.args.addtable)
        status = self.conn.tableExists(self.model_table_name)
        if status.exists:
            self.model_table = self.conn.CASTable(self.model_table_name)
        else:
            raise DLPyError("Failed to upload client side model to CAS server")
