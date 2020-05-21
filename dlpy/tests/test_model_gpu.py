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
import dlpy
from swat.cas.table import CASTable
from dlpy.model import Model, Optimizer, AdamSolver, Sequence, Gpu
from dlpy.sequential import Sequential
from dlpy.timeseries import TimeseriesTable
from dlpy.layers import (InputLayer, Conv2d, Conv1d, Pooling, Dense, OutputLayer,
                         Recurrent, Keypoints, BN, Res, Concat, Reshape, GlobalAveragePooling1D)
from dlpy.utils import caslibify, caslibify_context, file_exist_on_server
from dlpy.applications import Tiny_YoloV2
import unittest


class TestModel(unittest.TestCase):
    '''
    Please locate the images.sashdat file under the datasources to the DLPY_DATA_DIR.
    '''
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None
    data_dir_local = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.s)
        self.server_sep = '\\'
        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        if 'DLPY_DATA_DIR_LOCAL' in os.environ:
            self.data_dir_local = os.environ.get('DLPY_DATA_DIR_LOCAL')
            if self.data_dir_local.endswith(self.server_sep):
                self.data_dir_local = self.data_dir_local[:-1]
            self.data_dir_local += self.server_sep

    def CreateSimpleCNN1(self):
        model1 = Sequential(self.s, model_table='Simple_CNN1')
        model1.add(InputLayer(3, 224, 224))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Conv2d(8, 7))
        model1.add(Pooling(2))
        model1.add(Dense(16))
        model1.add(OutputLayer(act='softmax', n=2))        
        return model1

    def LoadEEE(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir+'images.sashdat', task='load')

        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)
        return caslib, tmp_caslib, 'eee', '_image_', '_label_'

    def test_model01(self):
        '''
        Enable GPU with all default values in Model.fit() and Model.predict().
            Gpu parameters object.

            Parameters
            ----------
            devices : list-of-ints, optional
                Specifies a list of GPU devices to be used.
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
            precision : string, optional
                Specifies the experimental option to incorporate lower computational
                precision in forward-backward computations to potentially engage tensor cores.
                Valid Values: FP32, FP16
                Default: FP32
            use_exclusive : bool, optional
                Specifies exclusive use of GPU devices.
                Default: False

            Returns
            -------
            :class:`Gpu`
        '''
        model1 = self.CreateSimpleCNN1()

        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=1)
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data=data, gpu=1)
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model02(self):
        '''
        gpu=Gpu(devices=[0]) in Model.fit() and Model.predict().
            devices : list-of-ints, optional
                Specifies a list of GPU devices to be used.
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(devices=[0]))
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data=data, gpu=Gpu(devices=[0]))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)
        
    def test_model03(self):
        '''
        gpu=Gpu(use_tensor_rt=True) in Model.fit() and Model.predict().
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(use_tensor_rt=True))
        #'WARNING: TensorRT only supports inference and will be disabled.'
        self.assertTrue(r.severity == 1)

        r2 = model1.predict(data=data, _debug=dict(display='d10c364.na.sas.com:0', ranks='0'), gpu=Gpu(use_tensor_rt=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)
        
    def test_model04(self):
        '''
        gpu=Gpu(precision='FP16') in Model.fit() and Model.predict().
            precision : string, optional
                Specifies the experimental option to incorporate lower computational
                precision in forward-backward computations to potentially engage tensor cores.
                Valid Values: FP32, FP16
                Default: FP32
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(precision='FP16'))
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data=data, gpu=Gpu(precision='FP16'))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)
                
    def test_model05(self):
        '''
        gpu=Gpu(use_exclusive=True) in Model.fit() and Model.predict().
            use_exclusive : bool, optional
                Specifies exclusive use of GPU devices.
                Default: False
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(use_exclusive=True))
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data=data, gpu=Gpu(use_exclusive=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model06(self):
        '''
        gpu=Gpu(devices=[0], use_tensor_rt=True) in Model.fit() and Model.predict().
            devices : list-of-ints, optional
                Specifies a list of GPU devices to be used.
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(devices=[0], use_tensor_rt=True))
        #'WARNING: TensorRT only supports inference and will be disabled.'
        self.assertTrue(r.severity == 1)

        r2 = model1.predict(data=data, gpu=Gpu(devices=[0], use_tensor_rt=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model07(self):
        '''
        gpu=Gpu(use_tensor_rt=True, precision='FP16') in Model.fit() and Model.predict().
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
            precision : string, optional
                Specifies the experimental option to incorporate lower computational
                precision in forward-backward computations to potentially engage tensor cores.
                Valid Values: FP32, FP16
                Default: FP32
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(use_tensor_rt=True, precision='FP16'))
        #'WARNING: TensorRT only supports inference and will be disabled.'
        self.assertTrue(r.severity == 1)

        r2 = model1.predict(data=data, gpu=Gpu(use_tensor_rt=True, precision='FP16'))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model08(self):
        '''
        gpu=Gpu(use_tensor_rt=True, use_exclusive=True) in Model.fit() and Model.predict().
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
            use_exclusive : bool, optional
                Specifies exclusive use of GPU devices.
                Default: False
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(use_tensor_rt=True, use_exclusive=True))
        #'WARNING: TensorRT only supports inference and will be disabled.'
        self.assertTrue(r.severity == 1)

        r2 = model1.predict(data=data, gpu=Gpu(use_tensor_rt=True, use_exclusive=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model09(self):
        '''
        gpu=Gpu(precision='FP16', use_exclusive=True) in Model.fit() and Model.predict().
            precision : string, optional
                Specifies the experimental option to incorporate lower computational
                precision in forward-backward computations to potentially engage tensor cores.
                Valid Values: FP32, FP16
                Default: FP32
            use_exclusive : bool, optional
                Specifies exclusive use of GPU devices.
                Default: False
        '''
        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(precision='FP16', use_exclusive=True))
        self.assertTrue(r.severity == 0)

        r2 = model1.predict(data=data, gpu=Gpu(precision='FP16', use_exclusive=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_model10(self):
        '''
        gpu=Gpu(use_tensor_rt=True, precision='FP16', use_exclusive=True) in Model.fit() and Model.predict().
            use_tensor_rt : bool, optional
                Enables using TensorRT for fast inference.
                Default: False.
            precision : string, optional
                Specifies the experimental option to incorporate lower computational
                precision in forward-backward computations to potentially engage tensor cores.
                Valid Values: FP32, FP16
                Default: FP32
            use_exclusive : bool, optional
                Specifies exclusive use of GPU devices.
                Default: False
        '''

        model1 = self.CreateSimpleCNN1()
        
        caslib, tmp_caslib, data, inputs, target = self.LoadEEE()

        r = model1.fit(data=data, inputs=inputs, target=target, gpu=Gpu(use_tensor_rt=True, precision='FP16', use_exclusive=True))
        #'WARNING: TensorRT only supports inference and will be disabled.'
        self.assertTrue(r.severity == 1)

        r2 = model1.predict(data=data, gpu=Gpu(use_tensor_rt=True, precision='FP16', use_exclusive=True))
        self.assertTrue(r2.severity == 0)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)
       
    def CreateYOLO(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path = self.data_dir + 'evaluate_obj_det_det.sashdat', task = 'load')

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'evaluate_obj_det_det', 'replace': True},
                               path = path)

        self.s.table.loadtable(caslib = caslib,
                               casout = {'name': 'evaluate_obj_det_gt', 'replace': True},
                               path = 'evaluate_obj_det_gt.sashdat')
        yolo_anchors = (5.9838598901098905,
                        3.4326923076923075,
                        2.184993862520458,
                        1.9841448445171848,
                        1.0261752136752136,
                        1.2277777777777779)
        yolo_model = Tiny_YoloV2(self.s, grid_number = 17, scale = 1.0 / 255,
                                 n_classes = 1, height = 544, width = 544,
                                 predictions_per_grid = 3,
                                 anchors = yolo_anchors,
                                 max_boxes = 100,
                                 coord_type = 'yolo',
                                 max_label_per_image = 100,
                                 class_scale = 1.0,
                                 coord_scale = 2.0,
                                 prediction_not_a_object_scale = 1,
                                 object_scale = 5,
                                 detection_threshold = 0.05,
                                 iou_threshold = 0.2)
        return yolo_model, 'evaluate_obj_det_gt', 'evaluate_obj_det_det'

    def test_evaluate_obj_det(self):
        yolo_model, ground_truth, detection_data = self.CreateYOLO()

        metrics = yolo_model.evaluate_object_detection(ground_truth = ground_truth, coord_type = 'yolo',
                                                       detection_data = detection_data, iou_thresholds=0.5,
                                                       gpu=1)

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level = 'error', caslib = caslib)
        
    def test_model_forecast1(self):        
        import datetime
        try:
            import pandas as pd
        except:
            unittest.TestCase.skipTest(self, "pandas not found in the libraries") 
        import numpy as np
            
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.table1 = TimeseriesTable.from_localfile(self.s, filename1, importoptions=importoptions1)
        self.table1.timeseries_formatting(timeid='datetime',
                                  timeseries=['series', 'covar'],
                                  timeid_informat='ANYDTDTM19.',
                                  timeid_format='DATETIME19.')
        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])
        self.table1.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series'],
                                         missing_handling='drop')
        
        valid_start = datetime.date(2015, 1, 4)
        test_start = datetime.date(2015, 1, 7)
        
        traintbl, validtbl, testtbl = self.table1.timeseries_partition(
                validation_start=valid_start, testing_start=test_start)
        
        model1 = Sequential(self.s, model_table='lstm_rnn')
        model1.add(InputLayer(std='STD'))
        model1.add(Recurrent(rnn_type='LSTM', output_type='encoding', n=15, reversed_=False))
        model1.add(OutputLayer(act='IDENTITY'))
        
        optimizer = Optimizer(algorithm=AdamSolver(learning_rate=0.01), mini_batch_size=32, 
                              seed=1234, max_epochs=10)                    
        seq_spec  = Sequence(**traintbl.sequence_opt)
        result = model1.fit(traintbl, valid_table=validtbl, optimizer=optimizer, gpu=1,
                            sequence=seq_spec, **traintbl.inputs_target)
        
        self.assertTrue(result.severity == 0)
        
        resulttbl1 = model1.forecast(horizon=1, gpu=1)
        self.assertTrue(isinstance(resulttbl1, CASTable))
        self.assertTrue(resulttbl1.shape[0]==15)
        
        local_resulttbl1 = resulttbl1.to_frame()
        unique_time = local_resulttbl1.datetime.unique()
        self.assertTrue(len(unique_time)==1)
        self.assertTrue(pd.Timestamp(unique_time[0])==datetime.datetime(2015,1,7))

        resulttbl2 = model1.forecast(horizon=3, gpu=1)
        self.assertTrue(isinstance(resulttbl2, CASTable))
        self.assertTrue(resulttbl2.shape[0]==45)
        
        local_resulttbl2 = resulttbl2.to_frame()
        local_resulttbl2.sort_values(by=['id1var', 'id2var', 'datetime'], inplace=True)
        unique_time = local_resulttbl2.datetime.unique()
        self.assertTrue(len(unique_time)==3)
        for i in range(3):
            self.assertTrue(pd.Timestamp(unique_time[i])==datetime.datetime(2015,1,7+i))
        
        series_lag1 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag1'].values
                                           
        series_lag2 = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             'series_lag2'].values
        
        DL_Pred = local_resulttbl2.loc[(local_resulttbl2.id1var==1) & (local_resulttbl2.id2var==1), 
                             '_DL_Pred_'].values
                                       
        self.assertTrue(np.array_equal(series_lag1[1:3], DL_Pred[0:2]))
        self.assertTrue(series_lag2[2]==DL_Pred[0])        

    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()


if __name__ == '__main__':
    unittest.main()
