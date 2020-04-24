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
import unittest
import swat
import swat.utils.testing as tm

from dlpy.images import ImageTable
from dlpy.utils import DLPyError
from dlpy.image_embedding import ImageEmbeddingTable


class TestImageEmbeddingTable(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    s = None
    server_sep = '/'
    data_dir = None

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

    def test_load_files(self):
        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        img_path = self.server_dir + 'DogBreed_small'
        my_images = ImageEmbeddingTable.load_files(self.s, path=img_path)
        print(my_images.columns)
        print(my_images.head())
        out_freq = my_images.freq(inputs='_dissimilar_')
        out_freq = out_freq['Frequency']
        print(out_freq)
        label, label1, label_pair, dissimilar = my_images.label_freq
        print(label)
        print(label1)
        print(label_pair)
        print(dissimilar)
        my_images.show()
        self.assertTrue(len(my_images) > 0)
        self.assertEqual(dissimilar['Frequency'][0], out_freq['Frequency'][0])

    def test_load_files_1(self):
        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        img_path = self.server_dir + 'DogBreed_small'
        my_images = ImageEmbeddingTable.load_files(self.s, path=img_path, n_samples=128)
        print(my_images.columns)
        print(my_images.head())
        out_freq = my_images.freq(inputs='_dissimilar_')
        out_freq = out_freq['Frequency']
        print(out_freq)
        label, label1, label_pair, dissimilar = my_images.label_freq
        print(label)
        print(label1)
        print(label_pair)
        print(dissimilar)
        my_images.show(randomize=True, n_image_pairs=10)
        self.assertTrue(len(my_images) > 0)
        self.assertEqual(dissimilar['Frequency'][0], out_freq['Frequency'][0])

    def test_load_files_resize(self):
        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        img_path = self.server_dir + 'DogBreed_small'
        my_images = ImageEmbeddingTable.load_files(self.s, path=img_path, n_samples=128,
                                                   resize_width=224, resize_height=224)
        print(my_images.columns)
        print(my_images.head())
        out_freq = my_images.freq(inputs='_dissimilar_')
        out_freq = out_freq['Frequency']
        print(out_freq)
        label, label1, label_pair, dissimilar = my_images.label_freq
        print(label)
        print(label1)
        print(label_pair)
        print(dissimilar)
        my_images.show(randomize=True, n_image_pairs=10)
        self.assertTrue(len(my_images) > 0)
        self.assertEqual(dissimilar['Frequency'][0], out_freq['Frequency'][0])

    def test_load_files_triplet(self):
        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        img_path = self.server_dir + 'DogBreed_small'
        my_images = ImageEmbeddingTable.load_files(self.s, path=img_path, embedding_model_type='triplet')
        print(my_images.columns)
        print(my_images.head())
        label, label1, label2, label_triplet = my_images.label_freq
        print(label)
        print(label1)
        print(label2)
        print(label_triplet)
        my_images.show(randomize=True, n_image_pairs=2)
        self.assertTrue(len(my_images) > 0)
        self.assertTrue(label_triplet['Frequency'][0] > 0)

    def test_load_files_quartet(self):
        if self.server_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_SERVER is not set in the environment variables")

        img_path = self.server_dir + 'DogBreed_small'
        my_images = ImageEmbeddingTable.load_files(self.s, path=img_path, embedding_model_type='quartet', n_samples=64)
        print(my_images.columns)
        print(my_images.head())
        label, label1, label2, label3, label_quartet = my_images.label_freq
        print(label)
        print(label1)
        print(label2)
        print(label3)
        print(label_quartet)
        print(label_quartet['Frequency'][0])
        my_images.show(randomize=True, n_image_pairs=2)
        self.assertTrue(len(my_images) > 0)
        self.assertTrue(label_quartet['Frequency'][0] > 0)


if __name__ == '__main__':
    unittest.main()
