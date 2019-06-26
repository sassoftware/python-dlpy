from speech import *
import unittest
import swat
import swat.utils.testing as tm
import wave
import tempfile


class TestSpeechUtils(unittest.TestCase):
    def setUp(self):
        self.data_dir_local = None
        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            self.data_dir_local = os.environ.get("DLPY_DATA_DIR_LOCAL")

    def test_read_audio_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(wave.Error):
            read_audio(os.path.join(self.data_dir_local, "acoustic_model_cpu.sashdat"))
        with self.assertRaises(wave.Error):
            read_audio(os.path.join(self.data_dir_local, "language_model.csv"))

    def test_read_audio_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(FileNotFoundError):
            read_audio(os.path.join(self.data_dir_local, "nonexistent.wav"))

    def test_read_audio_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertIsInstance(wave_reader, wave.Wave_read)
        self.assertIsInstance(wave_params, tuple)
        self.assertIsNotNone(wave_reader)
        self.assertIsNotNone(wave_params)
        wave_reader.close()

    def test_check_framerate_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_8khz.wav"))
        self.assertFalse(check_framerate(wave_params, 16000))
        wave_reader.close()
        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertTrue(check_framerate(wave_params, 16000))
        wave_reader.close()
        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_44khz.wav"))
        self.assertFalse(check_framerate(wave_params, 16000))
        wave_reader.close()

    def test_check_sampwidth_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_8bit_16khz.wav"))
        self.assertFalse(check_sampwidth(wave_params, 2))
        wave_reader.close()
        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        self.assertTrue(check_sampwidth(wave_params, 2))
        wave_reader.close()

    def test_check_sampwidth_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        with self.assertRaises(DLPyError):
            check_sampwidth(wave_params, 2, sampwidth_options=(3, 4))
        wave_reader.close()

    def test_convert_framerate_1(self):
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
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16k to 8k
        new_fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                         wave_params.framerate, wave_params.framerate // 2)
        self.assertEqual(len(fragment) / 2, len(new_fragment))
        self.assertEqual(b"".join([fragment[i: i + 4] for i in range(0, 200, 8)]), new_fragment[:100])
        wave_reader.close()

    def test_convert_framerate_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16k to 32k
        new_fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                         wave_params.framerate, wave_params.framerate * 2)
        self.assertAlmostEqual(len(fragment) / len(new_fragment), 0.5, 2)
        wave_reader.close()

    def test_convert_sampwidth_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16 bit to 16 bit
        new_fragment = convert_sampwidth(fragment, wave_params.sampwidth, wave_params.sampwidth)
        self.assertEqual(fragment, new_fragment)
        wave_reader.close()

    def test_convert_sampwidth_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader_8bit, wave_params_8bit = read_audio(os.path.join(self.data_dir_local, "sample_8bit_16khz.wav"))
        fragment_8bit = wave_reader_8bit.readframes(1000)
        # convert from 8 bit to 16 bit
        fragment_8bit_to_16bit = convert_sampwidth(fragment_8bit,
                                                   wave_params_8bit.sampwidth, wave_params_8bit.sampwidth * 2)
        self.assertEqual(len(fragment_8bit), 0.5 * len(fragment_8bit_to_16bit))
        wave_reader_8bit.close()

        # compare with the 16 bit sample file
        wave_reader_16bit, wave_params_16bit = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment_16bit = wave_reader_16bit.readframes(1000)
        self.assertEqual(len(fragment_8bit_to_16bit), len(fragment_16bit))
        wave_reader_16bit.close()

    def test_convert_sampwidth_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        wave_reader, wave_params = read_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"))
        fragment = wave_reader.readframes(1000)
        # convert from 16 bit to 8 bit
        new_fragment = convert_sampwidth(fragment, wave_params.sampwidth, wave_params.sampwidth // 2)
        self.assertEqual(len(fragment), len(new_fragment) * 2)
        wave_reader.close()

    def test_segment_audio_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            with self.assertRaises(DLPyError):
                segment_audio(os.path.join(self.data_dir_local, "nonexistent.wav"),
                              temp_dir, data_path_after_caslib, 20, 16000, 2)

    def test_segment_audio_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 20, 16000, 2)

            self.assertIsNotNone(listing_path_after_caslib)
            self.assertTrue(os.path.exists(listing_path_local))
            with open(listing_path_local, "r") as listing_file:
                lines = listing_file.readlines()
                self.assertEqual(len(lines), 1)
            self.assertEqual(len(segment_path_after_caslib_list), 1)
            self.assertEqual(len(segment_path_local_list), 1)
            clean_audio(listing_path_local, segment_path_local_list)

    def test_segment_audio_3(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 2, 16000, 2)

            self.assertIsNotNone(listing_path_after_caslib)
            self.assertTrue(os.path.exists(listing_path_local))
            with open(listing_path_local, "r") as listing_file:
                lines = listing_file.readlines()
                self.assertEqual(len(lines), 4)
            self.assertEqual(len(segment_path_after_caslib_list), 4)
            self.assertEqual(len(segment_path_local_list), 4)
            clean_audio(listing_path_local, segment_path_local_list)

    def test_clean_audio_1(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 2, 16000, 2)

            self.assertTrue(os.path.exists(listing_path_local))
            clean_audio(listing_path_local, segment_path_local_list)
            self.assertFalse(os.path.exists(listing_path_local))

    def test_clean_audio_2(self):
        if self.data_dir_local is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with tempfile.TemporaryDirectory() as temp_dir:
            data_path_after_caslib = "test/"
            listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
                segment_audio(os.path.join(self.data_dir_local, "sample_16bit_16khz.wav"),
                              temp_dir, data_path_after_caslib, 2, 16000, 2)

            for segment_path_local in segment_path_local_list:
                self.assertTrue(os.path.exists(segment_path_local))
            clean_audio(listing_path_local, segment_path_local_list)
            for segment_path_local in segment_path_local_list:
                self.assertFalse(os.path.exists(segment_path_local))


class TestSpeechToTextInit(unittest.TestCase):
    conn = None
    caslib_name = "caslib_test"
    data_path_after_caslib = None
    local_path = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.conn = swat.CAS()
        server_type = tm.get_cas_host_type(self.conn)
        if server_type.startswith("lin") or server_type.startswith("osx"):
            server_sep = "/"
        else:
            server_sep = "\\"

        if "DLPY_DATA_DIR" in os.environ:
            data_dir = os.environ.get("DLPY_DATA_DIR")
            if data_dir.endswith(server_sep):
                data_dir = data_dir[:-1]

            caslib_path, self.data_path_after_caslib = os.path.split(data_dir)
            if not caslib_path.endswith(server_sep):
                caslib_path += server_sep
            if not self.data_path_after_caslib.endswith(server_sep):
                self.data_path_after_caslib += server_sep

            self.conn.addCaslib(name=self.caslib_name, path=caslib_path,
                                dataSource=dict(srcType="PATH"), subDirectories=True)

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            self.local_path = os.environ.get("DLPY_DATA_DIR_LOCAL")

    def test_init_1(self):
        SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        action_set_list = self.conn.actionSetInfo().setinfo["actionset"].tolist()
        self.assertTrue("audio" in action_set_list)
        self.assertTrue("deepLearn" in action_set_list)
        self.assertTrue("langModel" in action_set_list)

    def test_init_2(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        SpeechToText(self.conn, self.data_path_after_caslib, self.local_path,
                     acoustic_model_path=self.data_path_after_caslib + "acoustic_model_cpu.sashdat",
                     language_model_path=self.data_path_after_caslib + "language_model.csv")
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("ASR" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS_ATTR" in table_list)
        self.assertTrue("LM" in table_list)

    def test_load_acoustic_model_1(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        speech.load_acoustic_model(self.data_path_after_caslib + "acoustic_model_cpu.sashdat")
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("ASR" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS_ATTR" in table_list)

    def test_load_acoustic_model_2(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(DLPyError):
            speech.load_acoustic_model(self.data_path_after_caslib + "acoustic_model_nonexistent.sashdat")

    def test_load_acoustic_model_3(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        speech.load_acoustic_model(
            self.data_path_after_caslib + "acoustic_model_cpu.sashdat",
            model_weights_path=self.data_path_after_caslib + "acoustic_model_cpu_weights.sashdat",
            model_weights_attr_path=self.data_path_after_caslib + "acoustic_model_cpu_weights_attr.sashdat")
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("ASR" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS_ATTR" in table_list)

    def test_load_acoustic_model_4(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        speech.load_acoustic_model(
            self.data_path_after_caslib + "acoustic_model_cpu.sashdat",
            model_caslib=self.caslib_name,
            model_weights_path=self.data_path_after_caslib + "acoustic_model_cpu_weights.sashdat",
            model_weights_caslib=self.caslib_name,
            model_weights_attr_path=self.data_path_after_caslib + "acoustic_model_cpu_weights_attr.sashdat",
            model_weights_attr_caslib=self.caslib_name)
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("ASR" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS" in table_list)
        self.assertTrue("PRETRAINED_WEIGHTS_ATTR" in table_list)

    def test_load_language_model_1(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        speech.load_language_model(self.data_path_after_caslib + "language_model.csv")
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("LM" in table_list)

    def test_load_language_model_2(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(DLPyError):
            speech.load_language_model(self.data_path_after_caslib + "language_model_nonexistent.csv")

    def test_load_language_model_3(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        speech = SpeechToText(self.conn, self.data_path_after_caslib, self.local_path, self.caslib_name)
        with self.assertRaises(AttributeError):
            self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()

        speech.load_language_model(self.data_path_after_caslib + "language_model.csv", model_caslib=self.caslib_name)
        table_list = self.conn.tableInfo(caslib=self.caslib_name).TableInfo["Name"].tolist()
        self.assertTrue("LM" in table_list)

    def tearDown(self):
        try:
            self.conn.terminate()
        except swat.SWATError:
            pass
        del self.conn
        swat.reset_option()


class TestSpeechToText(unittest.TestCase):
    conn = None
    caslib_name = "caslib_test"
    data_path_after_caslib = None
    local_path = None

    listing_path_local = None
    segment_path_list = None
    segment_path_local_list = None

    @classmethod
    def setUpClass(cls):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        cls.conn = swat.CAS()

        server_type = tm.get_cas_host_type(cls.conn)
        if server_type.startswith("lin") or server_type.startswith("osx"):
            server_sep = "/"
        else:
            server_sep = "\\"

        if "DLPY_DATA_DIR" in os.environ:
            data_dir = os.environ.get("DLPY_DATA_DIR")
            if data_dir.endswith(server_sep):
                data_dir = data_dir[:-1]

            caslib_path, cls.data_path_after_caslib = os.path.split(data_dir)
            if not caslib_path.endswith(server_sep):
                caslib_path += server_sep
            if not cls.data_path_after_caslib.endswith(server_sep):
                cls.data_path_after_caslib += server_sep

            cls.conn.addCaslib(name=cls.caslib_name, path=caslib_path,
                               dataSource=dict(srcType="PATH"), subDirectories=True)

        if "DLPY_DATA_DIR_LOCAL" in os.environ:
            cls.local_path = os.environ.get("DLPY_DATA_DIR_LOCAL")

        if "DLPY_DATA_DIR" in os.environ and "DLPY_DATA_DIR_LOCAL" in os.environ:
            cls.speech = SpeechToText(
                cls.conn, cls.data_path_after_caslib, cls.local_path,
                acoustic_model_path=cls.data_path_after_caslib + "acoustic_model_cpu.sashdat",
                acoustic_model_weights_path=cls.data_path_after_caslib + "acoustic_model_cpu_weights.sashdat",
                acoustic_model_weights_attr_path=cls.data_path_after_caslib + "acoustic_model_cpu_weights_attr.sashdat",
                language_model_path=cls.data_path_after_caslib + "language_model.sashdat")

    def step_1_load_audio(self):
        self.listing_path_local, self.segment_path_list, self.segment_path_local_list = \
            load_audio(self.speech,
                       audio_path=os.path.join(self.local_path, "sample_16bit_16khz.wav"),
                       casout=dict(name="audio", replace=True), segment_len=2)

        self.assertEqual(len(self.segment_path_list), 4)
        self.assertEqual(len(self.segment_path_local_list), 4)
        with open(self.listing_path_local, "r") as listing_file:
            lines = listing_file.readlines()
            self.assertEqual(len(lines), 4)

    def step_2_extract_acoustic_features(self):
        extract_acoustic_features(self.speech,
                                  audio_table="audio",
                                  casout=dict(name="feature", replace=True))
        num_of_rows = self.conn.tableInfo(name="feature").TableInfo["Rows"][0]
        self.assertEqual(num_of_rows, 4)

    def step_3_score_acoustic_features_1(self):
        with self.assertRaises(DLPyError):
            score_acoustic_features(self.speech,
                                    feature_table="feature",
                                    casout=dict(name="score", replace=True), gpu_devices={0})

    def step_3_score_acoustic_features_2(self):
        score_acoustic_features(self.speech,
                                feature_table="feature",
                                casout=dict(name="score", replace=True))
        num_of_rows = self.conn.tableInfo(name="score").TableInfo["Rows"][0]
        self.assertEqual(num_of_rows, 4)

    def step_4_decode_scores(self):
        decode_scores(self.speech,
                      score_table="score",
                      casout=dict(name="result", replace=True))
        num_of_rows = self.conn.tableInfo(name="result").TableInfo["Rows"][0]
        self.assertEqual(num_of_rows, 4)

    def step_5_concatenate_results(self):
        result = concatenate_results(self.speech,
                                     result_table_name="result",
                                     segment_path_list=self.segment_path_list)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_steps(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        self.step_1_load_audio()
        self.step_2_extract_acoustic_features()
        self.step_3_score_acoustic_features_1()
        self.step_3_score_acoustic_features_2()
        self.step_4_decode_scores()
        self.step_5_concatenate_results()

        clean_audio(self.listing_path_local, self.segment_path_local_list)

    def test_transcribe_1(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_path, "sample_16bit_16khz.wav"))
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_transcribe_2(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_path, "sample_16bit_16khz.wav"), segment_len=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_transcribe_3(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_path, "sample_8bit_16khz.wav"), segment_len=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_transcribe_4(self):
        if self.data_path_after_caslib is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables.")
        if self.local_path is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR_LOCAL is not set in the environment variables.")

        result = self.speech.transcribe(os.path.join(self.local_path, "sample_16bit_44khz.wav"), segment_len=5)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.conn.terminate()
        except swat.SWATError:
            pass
        del cls.conn
        swat.reset_option()
