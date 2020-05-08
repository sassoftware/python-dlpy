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
    s = None
    server_sep = '/'
    data_dir = None
    local_dir = None
    server_dir = None

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

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            self.local_dir = os.environ.get("DLPY_DATA_DIR_LOCAL")

        # the server path that points to DLPY_DATA_DIR_LOCAL
        if "DLPY_DATA_DIR_SERVER" in os.environ:
            self.server_dir = os.environ.get("DLPY_DATA_DIR_SERVER")
            if self.server_dir.endswith(self.server_sep):
                self.server_dir = self.server_dir[:-1]
            self.server_dir += self.server_sep

    def tearDown(self):
        # tear down tests
        try:
            self.s.terminate()
        except swat.SWATError:
            pass
        del self.s
        swat.reset_option()

    def test_audio_1(self):
        self.assertTrue(AudioTable.load_audio_files(self.s, "/u/") is None)

    def test_audio_11(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        print(self.data_dir+'listingFile.txt')

        self.assertTrue(AudioTable.load_audio_files(self.s, self.data_dir+'listingFile.txt',
                                                    self.s.CASTable('hebele')) is not None)

    def test_audio_12(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        with self.assertRaises(DLPyError):
            AudioTable.load_audio_files(self.s, self.data_dir.join('listingFile.txt'),
                                        caslib=self.s.caslibinfo().CASLibInfo['Name'][0])

    def test_audio_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        self.assertTrue(AudioTable.load_audio_files(self.s, self.data_dir+'listingFile.txt') is not None)

    def test_audio_4(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.s, self.data_dir+'listingFile.txt')
        if audio_table is not None:
            self.assertTrue(AudioTable.extract_audio_features(self.s, audio_table) is not None)
        else:
            self.assertTrue(False)

    def test_audio_44(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.s, self.data_dir+'listingFile.txt')
        if audio_table is not None:
            self.assertTrue(AudioTable.extract_audio_features(self.s, audio_table,
                                                              casout=self.s.CASTable('name')) is not None)
        else:
            self.assertTrue(False)

    def test_audio_5(self):
        self.assertTrue(AudioTable.extract_audio_features(self.s, 'string') is None)

    def test_audio_5(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.load_audio_files(self.s, self.data_dir+'listingFile.txt')
        fe = self.s.fetch(audio_table)
        self.assertTrue('recording' in fe.Fetch.ix[0]['_path_'])

    def test_audio_6(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)
        metadata_table = AudioTable.load_audio_metadata(self.s,
                                                        full_filename,
                                                        audio_path=project_path)
        fe = self.s.fetch(metadata_table)

    def test_audio_7(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.create_audio_table(self.s,
                                                    self.data_dir+'listingFile.txt',
                                                    full_filename)
        self.assertTrue(audio_table is not None)

    def test_audio_71(self):
        filename = os.path.join('datasources', 'metadata_for_audio.txt')
        project_path = os.path.dirname(os.path.abspath(__file__))
        full_filename = os.path.join(project_path, filename)

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        audio_table = AudioTable.create_audio_table(self.s, self.data_dir+'listingFile.txt',
                                                    full_filename, casout=self.s.CASTable('name99'))
        self.assertTrue(audio_table is not None)

    def test_audio_local_audio_path(self):

        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        try:
            import soundfile as sf
            import sounddevice as sd
        except (ModuleNotFoundError, ImportError):
            unittest.TestCase.skipTest(self, "skipping, soundfile and sounddevice packages are not installed")

        local_audio_dir = os.path.join(self.local_dir, 'lang_id', 'train')
        server_audio_dir = self.server_dir + 'lang_id' + '/train'

        audio_table = AudioTable.load_audio_files(self.s,
                                                  local_audio_path=local_audio_dir,
                                                  server_audio_path=server_audio_dir)
        numrows = audio_table.numrows()
        print(numrows['numrows'])
        self.assertEqual(numrows['numrows'], 15)

        # test extracting features
        feature_table = audio_table.create_audio_feature_table()
        print(feature_table.summary())

        # test extracting features with labels
        feature_table = audio_table.create_audio_feature_table(label_level=-2)
        print(feature_table.freq(inputs='_label_'))
        print(feature_table.label_freq)
        print(feature_table.feature_vars)
        self.assertEqual(feature_table.feature_vars[0], '_f0_v0_')

    def test_audio_local_audio_path_specgram(self):

        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        try:
            import soundfile as sf
            import sounddevice as sd
        except (ModuleNotFoundError, ImportError):
            unittest.TestCase.skipTest(self, "skipping, soundfile and sounddevice packages are not installed")

        local_audio_dir = os.path.join(self.local_dir, 'lang_id', 'train')
        server_audio_dir = self.server_dir + 'lang_id' + '/train'

        image_table = AudioTable.load_audio_files(self.s,
                                                  local_audio_path=local_audio_dir,
                                                  server_audio_path=server_audio_dir,
                                                  as_specgram=True)
        numrows = image_table.numrows()
        print(numrows['numrows'])
        self.assertEqual(numrows['numrows'], 15)

        print(image_table.columns)

        image_table.show(id='_path_', ncol=2, nimages=2)

    def test_audio_local_audio_path_specgram_label_level(self):

        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables")

        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        try:
            import soundfile as sf
            import sounddevice as sd
        except (ModuleNotFoundError, ImportError):
            unittest.TestCase.skipTest(self, "skipping, soundfile and sounddevice packages are not installed")

        local_audio_dir = os.path.join(self.local_dir, 'lang_id', 'train')
        server_audio_dir = self.server_dir + 'lang_id' + '/train'

        image_table = AudioTable.load_audio_files(self.s,
                                                  local_audio_path=local_audio_dir,
                                                  server_audio_path=server_audio_dir,
                                                  as_specgram=True, label_level=-2)
        numrows = image_table.numrows()
        print(numrows['numrows'])
        self.assertEqual(numrows['numrows'], 15)

        print(image_table.columns)

        print(image_table.label_freq)

        self.assertEqual(image_table.label_freq['Frequency'][0], 5)

        image_table.show(id='_path_', ncol=2, nimages=2)


if __name__ == '__main__':
    unittest.main()
