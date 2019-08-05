from dlpy.speech_utils import *
from dlpy.audio import AudioTable
from dlpy.model import Model
from dlpy.utils import get_server_path_sep, get_cas_host_type, caslibify
import os
import platform


class Speech:
    """
    Class to Do Speech Recognition Using SAS Viya.

    Parameters
    ----------
    conn : CAS Connection
        Specifies the CAS connection object

    data_path : string
        Specifies the absolute path of the folder where segmented audio files are stored (server side).

        The "audio_path" parameter in "transcribe" method is located on the client side. To transcribe the audio,
        we need to firstly save the .wav file somewhere the CAS server could access. Also, if the audio lasts
        very long, we may need to segment it into multiple files before copying.

        Notice that this is the location to store the temporary audio files. The Python client should have both
        reading and writing permission of this folder, and the CAS server should have at least reading permission
        of this folder.

    local_path : string, optional
        Specifies the path of the folder where segmented audio files are stored (client side).
        Default = None

        Notice that "data_path" and "local_path" actually point to the same location, and they should have
        the same value if the OS of the CAS server and the Python client are the same.

    acoustic_model_path : string, optional
        Specifies the absolute server-side path of the acoustic model file.
        Please make sure the weights file and the weights attribute file are placed under the same directory.
        Default = None

    language_model_path : string, optional
        Specifies the absolute server-side path of the language model file.
        Default = None
    """

    acoustic_model = None
    language_model_name = "languageModel"
    language_model_caslib = None

    data_path = None
    local_path = None

    data_caslib = None
    data_caslib_path = None
    data_path_after_caslib = None

    audio_table = None

    def __init__(self, conn,
                 data_path, local_path=None,
                 acoustic_model_path=None, language_model_path=None):
        try:
            import wave
        except ImportError:
            raise DLPyError("wave package is not found in the libraries. "
                            "Please install this package before using any APIs from dlpy.speech. "
                            "We're using this Python library to help read and write audio files.")
        try:
            import audioop
        except ImportError:
            raise DLPyError("audioop package is not found in the libraries. "
                            "Please install this package before using any APIs from dlpy.speech. "
                            "We're using this Python library to help extract audio features and convert audio formats.")

        self.conn = conn
        self.server_sep = get_server_path_sep(self.conn)

        self.data_path = data_path
        if self.data_path.endswith(self.server_sep):
            self.data_path = self.data_path[:-1]
        self.data_path += self.server_sep

        server_type = get_cas_host_type(self.conn).lower()
        is_server_unix = server_type.startswith("lin") or server_type.startswith("osx")

        client_type = platform.system()
        if (is_server_unix and client_type.startswith("Win")) or not (is_server_unix or client_type.startswith("Win")):
            if local_path is None:
                raise DLPyError("the \"local_path\" parameter is not specified. "
                                "The CAS server and the Python client have different OS type (Windows/Linux), "
                                "so please specify the \"local_path\" parameter.")
            else:
                self.local_path = local_path
        else:
            if local_path is None:
                self.local_path = self.data_path
                print("Note: the \"local_path\" parameter is not specified. "
                      "The CAS server and the Python client have the same OS type (Windows/Linux), "
                      "so simply use \"data_path\" as \"local_path\":", self.local_path)
            else:
                self.local_path = local_path

        if not os.path.exists(self.local_path):
            raise DLPyError("Invalid \"local_path\" value: does not exist.")
        if not os.access(self.local_path, os.R_OK):
            raise DLPyError("Invalid \"local_path\" value: does not have reading permission.")
        if not os.access(self.local_path, os.W_OK):
            raise DLPyError("Invalid \"local_path\" value: does not have writing permission.")

        self.conn.loadactionset("audio", _messagelevel="error")
        self.conn.loadactionset("deepLearn", _messagelevel="error")
        self.conn.loadactionset("langModel", _messagelevel="error")

        if acoustic_model_path is not None:
            self.load_acoustic_model(acoustic_model_path)

        if language_model_path is not None:
            self.load_language_model(language_model_path)

        self.data_caslib, self.data_path_after_caslib, _ = caslibify(self.conn, self.data_path, task="save")
        self.data_caslib_path = self.conn.caslibinfo(caslib=self.data_caslib).CASLibInfo["Path"][0]
        if not self.data_caslib_path.endswith(self.server_sep):
            self.data_caslib_path += self.server_sep

    def load_acoustic_model(self, acoustic_model_path):
        """
        Load the RNN acoustic model.

        Parameters
        ----------
        acoustic_model_path : string
            Specifies the absolute server-side path of the acoustic model file.
            Please make sure the weights file and the weights attribute file are placed under the same directory.

        """
        self.acoustic_model = Model(self.conn)
        self.acoustic_model.from_sashdat(self.conn, path=acoustic_model_path)
        if self.acoustic_model.model_table is None:
            raise DLPyError("Failed to load the acoustic model.")
        if self.acoustic_model.model_weights is None:
            raise DLPyError("Failed to load the acoustic model weights.")

    def load_language_model(self, language_model_path):
        """
        Load the N-gram language model.

        Parameters
        ----------
        language_model_path : string
            Specifies the absolute server-side path of the acoustic model file.

        """
        self.language_model_caslib, path_after_caslib, _ = caslibify(self.conn, language_model_path, task="load")
        rt = self.conn.retrieve("langModel.lmImport",
                                _messagelevel='error',
                                table=dict(name=path_after_caslib, caslib=self.language_model_caslib),
                                casout=dict(replace=True, name=self.language_model_name,
                                            caslib=self.language_model_caslib))
        if rt.severity > 1:
            self.language_model_caslib = None
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Failed to import the language model.")

    def transcribe(self, audio_path, gpu=None, max_path_size=100, alpha=1.0, beta=0.0):
        """
        Transcribe the audio file into text.

        Notice that for this API, we are assuming that the speech-to-test models published by SAS Viya 3.4 will be used.
        Please download the acoustic and language model files from here:
        https://support.sas.com/documentation/prod-p/vdmml/zip/speech_19w21.zip

        Parameters
        ----------
        audio_path : string
            Specifies the location of the audio file (client-side, absolute/relative).
        max_path_size : int, optional
            Specifies the maximum number of paths kept as candidates of the final results during the decoding process.
            Default = 100
        alpha : double, optional
            Specifies the weight of the language model, relative to the acoustic model.
            Default = 1.0
        beta : double, optional
            Specifies the weight of the sentence length, relative to the acoustic model.
            Default = 0.0
        gpu : class : `Gpu`, optional
            When specified, the action uses graphical processing unit hardware.
            The simplest way to use GPU processing is to specify "gpu=1". In this case, the default values of
            other GPU parameters are used.
            Setting gpu=1 enables all available GPU devices for use. Setting gpu=0 disables GPU processing.

        Returns
        -------
        string
        """

        # check if acoustic model is loaded
        if self.acoustic_model is None:
            raise DLPyError("acoustic model not found. "
                            "Please load the acoustic model by \"load_acoustic_model\" before calling \"transcribe\".")

        # check if language model is loaded
        if self.language_model_caslib is None:
            raise DLPyError("language model not found. "
                            "Please load the language model by \"load_language_model\" before calling \"transcribe\".")

        # step 1: preparation and segmentation
        listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
            segment_audio(audio_path, self.local_path, self.data_path_after_caslib, 10, 16000, 2)
        segment_path_list = [self.data_caslib_path + segment_path_after_caslib
                             for segment_path_after_caslib in segment_path_after_caslib_list]

        # step 2: load audio
        audio_table = AudioTable.load_audio_files(self.conn, path=listing_path_after_caslib, caslib=self.data_caslib)

        # step 3: extract features
        feature_table = AudioTable.extract_audio_features(self.conn, table=audio_table,
                                                          n_output_frames=3500, copyvars=["_path_"])

        # step 4: score features
        self.acoustic_model.score(table=feature_table,
                                  model="asr", init_weights="asr_weights", copy_vars=["_path_"], gpu=gpu,
                                  casout=dict(name="score_table", replace=True))
        score_table = self.conn.CASTable(name="score_table")

        # step 5: decode scores
        rt = self.conn.retrieve("langModel.lmDecode",
                                _messagelevel='error',
                                table=score_table,
                                casout=dict(name="result_table", replace=True),
                                langModelTable=dict(name=self.language_model_name, caslib=self.language_model_caslib),
                                blankLabel=" ",
                                spaceLabel="&",
                                maxPathSize=max_path_size,
                                alpha=alpha,
                                beta=beta,
                                copyvars=["_path_"])
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Failed to decode the scores.")
        result_table = self.conn.CASTable(name="result_table")

        # step 6: concatenate results
        result_dict = dict(zip(list(result_table["_path_"]), list(result_table["_audio_content_"])))
        result_list = [result_dict[segment_path] for segment_path in segment_path_list]
        result_list = [result.strip() for result in result_list]
        result_list = [result for result in result_list if len(result) > 0]
        result = " ".join(result_list)

        # step 7: cleaning
        clean_audio(listing_path_local, segment_path_local_list)

        return result
