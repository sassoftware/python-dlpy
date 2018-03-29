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
from dlpy.images import ImageTable

USER, PASSWD = tm.get_user_pass()
HOST, PORT, PROTOCOL = tm.get_host_port_proto()


class TestImageTable(tm.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.s = swat.CAS(HOST, PORT, USER, PASSWD, protocol=PROTOCOL)

        if type(self).server_type is None:
            # Set once per class and have every test use it. No need to change between tests.
            type(self).server_type = tm.get_cas_host_type(self.s)

        self.srcLib = tm.get_casout_lib(self.server_type)
        filename = os.path.join(os.path.dirname(__file__), 'datasources', 'ImageData.sashdat')
        r = tm.load_data(self.s, filename, self.server_type)

        self.tablename = r['tableName']
        self.assertNotEqual(self.tablename, None)
        self.table = ImageTable.from_table(r['casTable'])

    def tearDown(self):
        # tear down tests
        try:
            self.s.endsession()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()

    def test_table_type(self):
        self.assertTrue(isinstance(self.table, ImageTable))

    def test_colspec_str(self):
        dtype1 = 'varbinary'
        dtype2 = 'varchar'
        # if self.s._protocol in ['http', 'https']:
        #     dtype1 = 'binary'

        out = self.table.columninfo(inputs='_image_').ColumnInfo
        column_list = ['Column', 'ID', 'Type', 'RawLength', 'FormattedLength', 'NFL', 'NFD']
        value_list = ['_image_', 1, dtype1, 858354, 858354, 0, 0]
        for i, k in zip(column_list, value_list):
            self.assertEqual(out[i][0], k)

        out = self.table.columninfo(inputs='_label_').ColumnInfo
        column_list = ['Column', 'ID', 'Type', 'RawLength', 'FormattedLength', 'NFL', 'NFD']
        value_list = ['_label_', 2, dtype2, 7, 7, 0, 0]
        for i, k in zip(column_list, value_list):
            self.assertEqual(out[i][0], k)

        out = self.table.columninfo(inputs='_filename_0').ColumnInfo
        column_list = ['Column', 'ID', 'Type', 'RawLength', 'FormattedLength', 'NFL', 'NFD']
        value_list = ['_filename_0', 3, dtype2, 17, 17, 0, 0]
        for i, k in zip(column_list, value_list):
            self.assertEqual(out[i][0], k)

    def test_image_summary(self):
        out = self.table.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        value_list = [21, 500, 1024, 272, 1024, 840, 813, 134, 0, 255, 124, 0, 255, 87, 0, 255]
        for i, k in zip(column_list, value_list):
            self.assertEqual(int(out[i]), k)

    def test_resize(self):
        out = self.table.resize(width=200, height=200, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        value_list = [21, 200, 200, 200, 200, 200, 200, 134, 0, 255, 123, 0, 255, 87, 0, 255]
        for i, k in zip(column_list, value_list):
            self.assertEqual(int(out[i]), k)

    def test_crop(self):
        out = self.table.crop(x=0, y=0, width=200, height=200, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        value_list = [21, 200, 200, 200, 200, 200, 200, 137, 0, 255, 123, 0, 255, 86, 0, 255]
        for i, k in zip(column_list, value_list):
            self.assertEqual(int(out[i]), k)

    def test_as_patches(self):
        out = self.table.as_patches(x=0, y=0, width=200, height=200, step_size=100,
                                    output_width=100, output_height=100, inplace=False)
        out = out.image_summary
        column_list = ['jpg', 'minWidth', 'maxWidth', 'minHeight', 'maxHeight', 'meanWidth',
                       'meanHeight', 'mean1stChannel', 'min1stChannel', 'max1stChannel',
                       'mean2ndChannel', 'min2ndChannel', 'max2ndChannel', 'mean3rdChannel',
                       'min3rdChannel', 'max3rdChannel']
        value_list = [1312, 100, 100, 100, 100, 100, 100, 129, 0, 255, 125, 0, 255, 98, 0, 255]
        for i, k in zip(column_list, value_list):
            self.assertEqual(int(out[i]), k)


if __name__ == '__main__':
    tm.runtests()
