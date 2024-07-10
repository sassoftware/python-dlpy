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


import unittest
import swat
import swat.utils.testing as tm
import csv
import os
from pathlib import Path
from dlpy.mzmodel import *
from multiprocessing import Process

class TestModelzoo(unittest.TestCase):
    '''
        If you are using dlpy on a Windows machine, then copy datasources/dlpy_obj_det_test to both DLPY_DATA_DIR and
        DLPY_DATA_DIR_LOCAL. If you are using dlpy on a linux machine, then copy datasources/dlpy_obj_det_test to
        DLPY_DATA_DIR only.
        '''
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None
    data_dir_local = None
    code_cov_skip = 0

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
            if self.data_dir_local.startswith('/'):
                sep_ = '/'
            else:
                sep_ = '\\'
            if self.data_dir_local.endswith(sep_):
                self.data_dir_local = self.data_dir_local[:-1]
            self.data_dir_local += sep_

        if 'CODE_COV_SKIP' in os.environ:
            self.code_cov_skip = 1

    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()

    def test_mzmodel_train1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                        num_classes=10, model_path=self.data_dir + "resnet18.pt")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=1e-3, momentum=0.9),
                              batch_size=128,
                              max_epochs=3
                              )
        r = train1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_train2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'self_driving_256_test.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train2 = MZModel(conn=self.s, model_type="torchNative", model_name="enet", dataset_type='segmentation',
                         num_classes=13, model_path=self.data_dir + "enet_raw.pt")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=0.08),
                              batch_size=15,
                              max_epochs=3
                              )
        r = train2.train(table="eee", inputs='_image_', targets='labels', n_threads=5, optimizer=optimizer)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_train3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'self_driving_256_test.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train3 = MZModel(conn=self.s, model_type="torchScript", dataset_type='segmentation',
                         num_classes=13, model_path=self.data_dir + "deeplab_wrapped.pt")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=0.01),
                              batch_size=10,
                              max_epochs=3
                              )
        r = train3.train(table="eee", inputs='_image_', targets='labels', n_threads=5, optimizer=optimizer, gpu=[0])

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_train4(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'coco128/obj_table.txt', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train4 = MZModel(conn=self.s, model_type="torchNative", model_name="yolov5", model_subtype='small',
                         dataset_type='objdetect', num_classes=80,
                         anchors="10 13 16 30 33 23 30 61 62 45 59 119 116 90 156 198 373 326",
                         model_path=self.data_dir + "coco128/traced_yolov5s.pt")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=0.08),
                              batch_size=15,
                              max_epochs=3
                              )
        r = train4.train(table="eee", inputs='img_path', targets='label_path', n_threads=5, optimizer=optimizer)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_train5(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s,
                                             path=self.data_dir + 'fashion_mnist_valid.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train5 = MZModel(conn=self.s, model_type="torchNative", model_name="DRNN",
                         num_classes=10, model_path=self.data_dir + "drnn_classifier_model.pt",
                         input_size=32, hidden_size=20, num_layers=2,
                         rnn_type="RNN")

        train5.add_image_transformation(image_size='32 32')
        train5.add_text_transformation(word_embedding='word2Vec')

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=1e-3, momentum=0.9),
                              batch_size=128,
                              max_epochs=3
                              )
        r = train5.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_score1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        score1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                        num_classes=10, model_path=self.data_dir + "resnet18.pt")

        r = score1.score(table="eee", inputs="_image_", targets="xlabels", batch_size=128)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_score2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'self_driving_256_test.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        score2 = MZModel(conn=self.s, model_type="torchNative", model_name="enet", dataset_type='segmentation',
                         num_classes=13, model_path=self.data_dir + "enet_raw.pt")

        r = score2.score(table="eee", inputs='_image_', targets='labels', batch_size=15)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_score3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'self_driving_256_test.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        score3 = MZModel(conn=self.s, model_type="torchScript", dataset_type='segmentation',
                         num_classes=13, model_path=self.data_dir + "deeplab_wrapped.pt")

        r = score3.score(table="eee", inputs='_image_', targets='labels', batch_size=10)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_score4(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'coco128/obj_table.txt', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        score4 = MZModel(conn=self.s, model_type="torchNative", model_name="yolov5", model_subtype='small',
                         dataset_type='objdetect', num_classes=80,
                         anchors="10 13 16 30 33 23 30 61 62 45 59 119 116 90 156 198 373 326",
                         model_path=self.data_dir + "coco128/traced_yolov5s.pt")

        r = score4.score(table="eee", inputs='img_path', targets='label_path', batch_size=15)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_save_to_astore(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                         num_classes=10, model_path=self.data_dir + "resnet18.pt")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=1e-3, momentum=0.9),
                              batch_size=128,
                              max_epochs=3
                              )
        r = train1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer)

        self.assertTrue(r.severity == 0)

        with self.assertRaises(DLPyError):
            train1.save_to_astore(path='/amran/komran')

    def test_mzmodel_tune1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                        num_classes=10, model_path=self.data_dir + "resnet18.pt")

        lr = HyperRange(lower=5e-4, upper=1e-3)
        batch_size = BatchSizeRange(lower=100, upper=150)
        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=lr, momentum=0.9),
                              batch_size=batch_size,
                              max_epochs=10
                              )
        tuner = Tuner(method="EAGLS", pop_size=5, max_func=5)

        r = train1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer, tuner=tuner)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_from_client(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        model1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18", num_classes=10)
        print(model1.documents_train)

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=1e-3, momentum=0.9),
                              batch_size=128,
                              max_epochs=3
                              )
        r = model1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer, log_level=2)
        print(r)

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))
        
        # should print out warning message
        # model1.upload_model_from_client(Path(self.data_dir_local) / "resnet18.pt")
        self.s.droptable(model1.model_table_name)
        model1.upload_model_from_client(Path(self.data_dir_local) / "resnet18.pt")
        r = model1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer, log_level=2)
        print(r)

        if r.severity:
            raise DLPyError("WARNING or ERROR message shouldn't appear.")
        print(r)
        for msg in r.messages:
            print(msg)

        r = model1.score(table="eee", inputs="_image_", targets="xlabels", batch_size=128)
        for msg in r.messages:
            print(msg)
        
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_image_processing(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'self_driving_256_test.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train = MZModel(conn=self.s, model_type="torchNative", model_name="enet", dataset_type='segmentation',
                         num_classes=13, model_path=self.data_dir + "enet_raw.pt")

        train.add_image_transformation(image_size='256', image_resize_type="RETAIN_ASPECTRATIO", random_transform=True)

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=0.08),
                              batch_size=15,
                              max_epochs=3
                              )
        r = train.train(table="eee", inputs='_image_', targets='labels', n_threads=5, optimizer=optimizer, gpu=[0])

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)

    def test_mzmodel_image_processing2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'coco128/obj_table.txt', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        train = MZModel(conn=self.s, model_type="torchNative", model_name="yolov5", model_subtype='small',
                         dataset_type='objdetect', num_classes=80,
                         anchors="10 13 16 30 33 23 30 61 62 45 59 119 116 90 156 198 373 326",
                         model_path=self.data_dir + "coco128/traced_yolov5s.pt")

        train.add_image_transformation(image_size='640', image_resize_type="RETAIN_ASPECTRATIO", random_transform=True)

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=0.08),
                              batch_size=15,
                              max_epochs=3
                              )
        r = train.train(table="eee", inputs='img_path', targets='label_path', n_threads=5, optimizer=optimizer, gpu=[0])

        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)


