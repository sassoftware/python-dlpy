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
from dlpy.mzmodel_utils import *
from dlpy.mzmodel import *

class TestUtils(unittest.TestCase):
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

    def test_convert_torchvision_model_weights(self):
        cpp_model_path = self.data_dir_local + "resnet18_traced.pt"
        convert_torchvision_model_weights("resnet18", cpp_model_path)

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        caslib, path, tmp_caslib = caslibify(self.s, path=self.data_dir + 'cifar10_small.sashdat', task='load')
        self.s.table.loadtable(caslib=caslib,
                               casout={'name': 'eee', 'replace': True},
                               path=path)

        data_dir_server = os.environ.get('DLPY_DATA_DIR_SERVER')
        train1 = MZModel(conn=self.s, model_type="torchNative", model_name="resnet", model_subtype="resnet18",
                        num_classes=10, model_path=data_dir_server+"/resnet18_traced.pt")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        optimizer = Optimizer(seed=54321,
                              algorithm=SGDSolver(lr=1e-3, momentum=0.9),
                              batch_size=128,
                              max_epochs=3
                              )

        r = train1.train(table="eee", inputs="_image_", targets="xlabels", optimizer=optimizer)
        print(r)
        if r.severity > 0:
            for msg in r.messages:
                print(msg)
        self.assertLessEqual(r.severity, 1, msg="\n".join([msg for msg in r.messages]))

        if (caslib is not None) and tmp_caslib:
            self.s.retrieve('table.dropcaslib', message_level='error', caslib=caslib)



