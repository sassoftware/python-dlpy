from dlpy.speech import *
import unittest
import swat
import swat.utils.testing as tm
import tempfile
import os


class TestSpeechUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir_local = None
        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            self.data_dir_local = os.environ.get("DLPY_DATA_DIR_LOCAL")

    def test_read_audio_1(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(wave.Error):
            read_audio(os.path.join(self.data_dir_local, "sample_acoustic_model.sashdat"))
        with self.assertRaises(wave.Error):
            read_audio(os.path.join(self.data_dir_local, "sample_language_model.csv"))

    def test_read_audio_2(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(FileNotFoundError):
            read_audio(os.path.join(self.data_dir_local, "nonexistent.wav"))

    def test_read_audio_3(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertIsInstance(wave_reader, wave.Wave_read)
        self.assertIsInstance(wave_params, tuple)
        self.assertIsNotNone(wave_reader)
        self.assertIsNotNone(wave_params)
        wave_reader.close()

    def test_check_framerate(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_8khz.wav"))
        self.assertFalse(check_framerate(wave_params, 16000))
        wave_reader.close()
        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertTrue(check_framerate(wave_params, 16000))
        wave_reader.close()

    def test_check_sampwidth(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_8bit_16khz.wav"))
        self.assertFalse(check_sampwidth(wave_params, 2))
        wave_reader.close()
        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertTrue(check_sampwidth(wave_params, 2))
        wave_reader.close()

    def test_convert_framerate_1(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16k to 16k
        new_fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                         wave_params.framerate, wave_params.framerate)
        self.assertEqual(fragment, new_fragment)
        wave_reader.close()

    def test_convert_framerate_2(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16k to 8k
        new_fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                         wave_params.framerate, wave_params.framerate // 2)
        self.assertEqual(len(fragment) / 2, len(new_fragment))
        # convert from 16k to 32k
        new_fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                         wave_params.framerate, wave_params.framerate * 2)
        self.assertAlmostEqual(len(fragment) / len(new_fragment), 0.5, 2)
        wave_reader.close()

    def test_convert_sampwidth_1(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16 bit to 16 bit
        new_fragment = convert_sampwidth(fragment, wave_params.sampwidth, wave_params.sampwidth)
        self.assertEqual(fragment, new_fragment)
        wave_reader.close()

    def test_convert_sampwidth_2(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_8bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 8 bit to 16 bit
        new_fragment = convert_sampwidth(fragment, wave_params.sampwidth, wave_params.sampwidth * 2)
        self.assertEqual(len(fragment), 0.5 * len(new_fragment))
        wave_reader.close()

    def test_convert_sampwidth_3(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16 bit to 8 bit
        new_fragment = convert_sampwidth(fragment, wave_params.sampwidth, wave_params.sampwidth // 2)
        self.assertEqual(len(fragment), len(new_fragment) * 2)
        wave_reader.close()

    def test_segment_audio_1(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            with self.assertRaises(DLPyError):
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 40, 16000, 2)

    def test_segment_audio_2(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 20, 16000, 2)

            self.assertTrue(os.path.exists(listing_path_local))
            with open(listing_path_local, "r") as listing_file:
                lines = listing_file.readlines()
                self.assertEqual(len(lines), 1)
            self.assertEqual(len(segment_path_after_caslib_list), 1)
            self.assertEqual(len(segment_path_local_list), 1)
            clean_audio(listing_path_local, segment_path_local_list)

            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 2, 16000, 2)

            self.assertTrue(os.path.exists(listing_path_local))
            with open(listing_path_local, "r") as listing_file:
                lines = listing_file.readlines()
                self.assertEqual(len(lines), 4)
            self.assertEqual(len(segment_path_after_caslib_list), 4)
            self.assertEqual(len(segment_path_local_list), 4)
            clean_audio(listing_path_local, segment_path_local_list)

    def test_clean_audio(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 2, 16000, 2)

            self.assertTrue(os.path.exists(listing_path_local))
            for segment_path_local in segment_path_local_list:
                self.assertTrue(os.path.exists(segment_path_local))

            clean_audio(listing_path_local, segment_path_local_list)
            for segment_path_local in segment_path_local_list:
                self.assertFalse(os.path.exists(segment_path_local))


class TestSpeechToTextInit(unittest.TestCase):
    conn = None
    server_type = None
    server_sep = None

    data_dir = None
    local_dir = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.conn = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.conn)
        self.server_sep = "\\"
        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = "/"

        if "DLPY_DATA_DIR" in os.environ:
            self.data_dir = os.environ.get("DLPY_DATA_DIR")
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            self.local_dir = os.environ.get("DLPY_DATA_DIR_LOCAL")

    def test_init_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = Speech(self.conn, self.data_dir, self.local_dir)
        action_set_list = self.conn.actionSetInfo().setinfo["actionset"].tolist()
        self.assertTrue("audio" in action_set_list)
        self.assertTrue("deepLearn" in action_set_list)
        self.assertTrue("langModel" in action_set_list)
        self.assertIsNone(speech.acoustic_model)
        self.assertIsNone(speech.language_model_caslib)

    def test_init_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(DLPyError):
            Speech(self.conn, self.data_dir, os.path.join(self.local_dir, "nonexistent"))

    def test_init_3(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = Speech(self.conn, self.data_dir, self.local_dir,
                        self.data_dir + "sample_acoustic_model.sashdat",
                        self.data_dir + "sample_language_model.csv")
        self.assertIsNotNone(speech.acoustic_model)
        self.assertIsNotNone(speech.language_model_caslib)
        table_list = self.conn.tableInfo(caslib=speech.language_model_caslib).TableInfo["Name"].tolist()
        self.assertTrue(speech.language_model_name.upper() in table_list)

    def test_load_acoustic_model(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = Speech(self.conn, self.data_dir, self.local_dir)
        speech.load_acoustic_model(self.data_dir + "sample_acoustic_model.sashdat")
        self.assertIsNotNone(speech.acoustic_model)
        self.assertIsNotNone(speech.acoustic_model.model_name)
        self.assertIsNotNone(speech.acoustic_model.model_table)
        self.assertIsNotNone(speech.acoustic_model.model_weights)

    def test_load_language_model_1(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = Speech(self.conn, self.data_dir, self.local_dir)
        self.assertIsNone(speech.language_model_caslib)

        speech.load_language_model(self.data_dir + "sample_language_model.csv")
        self.assertIsNotNone(speech.language_model_caslib)

        table_list = self.conn.tableInfo(caslib=speech.language_model_caslib).TableInfo["Name"].tolist()
        self.assertTrue(speech.language_model_name.upper() in table_list)

    def test_load_language_model_2(self):
        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = Speech(self.conn, self.data_dir, self.local_dir)
        self.assertIsNone(speech.language_model_caslib)

        with self.assertRaises(DLPyError):
            speech.load_language_model(self.data_dir + "language_model_nonexistent.csv")
        self.assertIsNone(speech.language_model_caslib)

    def tearDown(self):
        try:
            self.conn.terminate()
        except swat.SWATError:
            pass
        del self.conn
        swat.reset_option()


class TestSpeechToText(unittest.TestCase):
    conn = None
    server_type = None
    server_sep = None

    data_dir = None
    local_dir = None

    speech = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.conn = swat.CAS()
        cls.server_type = tm.get_cas_host_type(cls.conn)
        cls.server_sep = "\\"
        if cls.server_type.startswith("lin") or cls.server_type.startswith("osx"):
            cls.server_sep = "/"

        if "DLPY_DATA_DIR" in os.environ:
            cls.data_dir = os.environ.get("DLPY_DATA_DIR")
            if cls.data_dir.endswith(cls.server_sep):
                cls.data_dir = cls.data_dir[:-1]
            cls.data_dir += cls.server_sep

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            cls.local_dir = os.environ.get("DLPY_DATA_DIR_LOCAL")

        if cls.data_dir is not None:
            cls.speech = Speech(cls.conn, cls.data_dir, cls.local_dir,
                                cls.data_dir + "sample_acoustic_model.sashdat",
                                cls.data_dir + "sample_language_model.csv")

    def test_transcribe_1(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_dir, "sample_16bit_16khz.wav"))
        self.assertIsInstance(result, str)

    def test_transcribe_2(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_dir, "sample_8bit_16khz.wav"))
        self.assertIsInstance(result, str)

    def test_transcribe_3(self):
        try:
            import wave
        except ImportError:
            unittest.TestCase.skipTest(self, "wave is not found in the libraries.")
        try:
            import audioop
        except ImportError:
            unittest.TestCase.skipTest(self, "audioop is not found in the libraries.")

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_dir, "sample_16bit_44khz.wav"))
        self.assertIsInstance(result, str)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.conn.terminate()
        except swat.SWATError:
            pass
        del cls.conn
        swat.reset_option()
