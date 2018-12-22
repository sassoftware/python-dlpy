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
#
import os
import unittest

import swat
import swat.utils.testing as tm

from dlpy.audio import AudioTable
from dlpy.utils import DLPyError


class TestAudioTable(unittest.TestCase):
    '''
    Please locate the listingFile.txt and three .wav files under your DLPY_DATA_DIR before running these examples
    '''

    server_type = None
    conn = None
    server_sep = '/'
    data_dir = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.conn = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.conn)

        cls.server_sep = '\\'
        if cls.server_type.startswith("lin") or cls.server_type.startswith("osx"):
            cls.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            cls.data_dir = os.environ.get('DLPY_DATA_DIR')
            if cls.data_dir.endswith(cls.server_sep):
                cls.data_dir = cls.data_dir[:-1]
            cls.data_dir += cls.server_sep

    @classmethod
    def tearDownClass(cls):
        # tear down tests
        try:
            cls.conn.terminate()
        except swat.SWATError:
            pass
        del cls.conn
        swat.reset_option()

    def test_audio_1(self):
        self.assertTrue(AudioTable.load_audio_files(self.conn, "/u/") is None)

    def test_audio_11(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        print(self.data_dir+'listingFile.txt')

        self.assertTrue(AudioTable.load_audio_files(self.conn, self.data_dir+'listingFile.txt',
                                                    self.conn.CASTable('hebele')) is not None)

    def test_audio_12(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        with self.assertRaises(DLPyError):
            AudioTable.load_audio_files(self.conn, self.data_dir.join('listingFile.txt'),
                                        caslib=self.conn.caslibinfo().CASLibInfo['Name'][0])

    def test_audio_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.assertTrue(AudioTable.load_audio_files(self.conn, self.data_dir+'listingFile.txt') is not None)

    def test_audio_4(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.conn, self.data_dir+'listingFile.txt')
        if audio_table is not None:
            self.assertTrue(AudioTable.extract_audio_features(self.conn, audio_table) is not None)
        else:
            self.assertTrue(False)

    def test_audio_44(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.conn, self.data_dir+'listingFile.txt')
        if audio_table is not None:
            self.assertTrue(AudioTable.extract_audio_features(self.conn, audio_table,
                                                              casout=self.conn.CASTable('name')) is not None)
        else:
            self.assertTrue(False)

    def test_audio_5(self):
        self.assertTrue(AudioTable.extract_audio_features(self.conn, 'string') is None)

    def test_audio_5(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.conn, self.data_dir+'listingFile.txt')
        fe = self.conn.fetch(audio_table)
        self.assertTrue('recording' in fe.Fetch.ix[0]['_path_'])

    def test_audio_6(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)
        metadata_table = AudioTable.load_audio_metadata(self.conn,
                                                        full_filename,
                                                        audio_path=project_path)
        fe = self.conn.fetch(metadata_table)

    def test_audio_7(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.create_audio_table(self.conn,
                                                    self.data_dir+'listingFile.txt',
                                                    full_filename)
        self.assertTrue(audio_table is not None)

    def test_audio_71(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.create_audio_table(self.conn, self.data_dir+'listingFile.txt',
                                                    full_filename, casout=self.conn.CASTable('name99'))
        self.assertTrue(audio_table is not None)




