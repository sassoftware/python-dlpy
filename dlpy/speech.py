from speech_utils import *
from dlpy.utils import get_server_path_sep
import os


class SpeechToText:
    acoustic_model_name = "asr"
    acoustic_model_weights_name = "pretrained_weights"
    acoustic_model_weights_attr_name = "pretrained_weights_attr"
    language_model_name = "lm"

    def __init__(self,
                 conn, data_path_after_caslib, local_path, data_caslib=None,
                 acoustic_model_path=None, acoustic_model_caslib=None,
                 acoustic_model_weights_path=None, acoustic_model_weights_caslib=None,
                 acoustic_model_weights_attr_path=None, acoustic_model_weights_attr_caslib=None,
                 language_model_path=None, language_model_caslib=None):
        """
        Initialize.

        Parameters
        ----------
        conn: CAS Connection
            Specifies the CAS connection object
        data_path_after_caslib: string
            Specifies a location where temporary files are stored (server-side, relative).
            Notice that the path should be accessible by CAS server and is relative to data_caslib.
        local_path: string
            Specifies a location where temporary audio files are generated (client-side, absolute/relative).
            Notice that data_path_after_caslib and local_path point to the same location.
        data_caslib: string, optional
            Specifies the caslib to load the temporary files.
            Notice that the current active caslib will be used if data_caslib is None.
            Default: None.
        acoustic_model_path: string, optional
            Specifies the path of the acoustic model table, relative to acoustic_model_caslib.
            Default: None
        acoustic_model_caslib: string, optional
            Specifies the caslib to load the acoustic model table.
            Default: None
        acoustic_model_weights_path: string, optional
            Specifies the path of the acoustic model weights table, relative to acoustic_model_weights_caslib.
            Default: None
        acoustic_model_weights_caslib: string, optional
            Specifies the caslib to load the acoustic model weights table.
            Default: None
        acoustic_model_weights_attr_path: string, optional
            Specifies the path of the acoustic model weights attribute table, relative to acoustic_model_weights_attr_caslib.
            Default: None
        acoustic_model_weights_attr_caslib: string, optional
            Specifies the caslib to load the acoustic model weights attribute table.
            Default: None
        language_model_path: string, optional
            Specifies the path of the language model table, relative to language_model_caslib.
            Default: None
        language_model_caslib: string, optional
            Specifies the caslib to load the language model table.
            Default: None

        """

        self.conn = conn
        self.server_sep = get_server_path_sep(self.conn)

        self.data_path_after_caslib = data_path_after_caslib
        if not self.data_path_after_caslib.endswith(self.server_sep):
            self.data_path_after_caslib += self.server_sep

        self.local_path = local_path
        if not os.path.exists(self.local_path):
            raise DLPyError("Invalid \"local_path\" value.")

        self.data_caslib = data_caslib

        self.conn.loadactionset("audio", _messagelevel="error")
        self.conn.loadactionset("deepLearn", _messagelevel="error")
        self.conn.loadactionset("langModel", _messagelevel="error")

        if acoustic_model_path is not None:
            self.load_acoustic_model(acoustic_model_path,
                                     model_caslib=acoustic_model_caslib,
                                     model_weights_path=acoustic_model_weights_path,
                                     model_weights_caslib=acoustic_model_weights_caslib,
                                     model_weights_attr_path=acoustic_model_weights_attr_path,
                                     model_weights_attr_caslib=acoustic_model_weights_attr_caslib)

        if language_model_path is not None:
            self.load_language_model(language_model_path,
                                     model_caslib=language_model_caslib)

    def load_acoustic_model(self,
                            model_path, model_caslib=None,
                            model_weights_path=None, model_weights_caslib=None,
                            model_weights_attr_path=None, model_weights_attr_caslib=None):
        """
        Load the RNN acoustic model from the existing tables.

        Parameters
        ----------
        model_path: string
            Specifies the path of the model table, relative to model_caslib.
        model_caslib: string, optional
            Specifies the caslib of the model table.
            Notice that the current active caslib will be used if model_caslib is None.
            Default: None
        model_weights_path: string, optional
            Specifies the path of the model weights table, relative to model_weights_caslib.
            Default: None
        model_weights_caslib: string, optional
            Specifies the caslib of the model weights table.
            Notice that the current active caslib will be used if model_weights_caslib is None.
            Default: None
        model_weights_attr_path: string, optional
            Specifies the path of the model weights attribute table, relative to model_weights_attr_caslib.
            Default: None
        model_weights_attr_caslib: string, optional
            Specifies the caslib of the model weights attribute table.
            Notice that the current active caslib will be used if model_weights_attr_caslib is None.
            Default: None

        """
        if model_weights_path is None:
            _file_name_, _extension_ = os.path.splitext(model_path)
            model_weights_path = _file_name_ + "_weights" + _extension_
            print("Note:", model_weights_path, "is used as the model weights table.")

        if model_weights_attr_path is None:
            _file_name_, _extension_ = os.path.splitext(model_weights_path)
            model_weights_attr_path = _file_name_ + "_attr" + _extension_
            print("Note:", model_weights_attr_path, "is used as the model weights attribute table.")

        if model_caslib is None:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    path=model_path,
                                    casout=dict(replace=True, name=self.acoustic_model_name))
        else:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    caslib=model_caslib,
                                    path=model_path,
                                    casout=dict(replace=True, name=self.acoustic_model_name))
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Cannot load the acoustic model.")

        if model_weights_caslib is None:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    path=model_weights_path,
                                    casout=dict(replace=True, name=self.acoustic_model_weights_name))
        else:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    caslib=model_weights_caslib,
                                    path=model_weights_path,
                                    casout=dict(replace=True, name=self.acoustic_model_weights_name))
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Cannot load the acoustic model weights.")

        if model_weights_attr_caslib is None:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    path=model_weights_attr_path,
                                    casout=dict(replace=True, name=self.acoustic_model_weights_attr_name))
        else:
            rt = self.conn.retrieve("table.loadTable",
                                    _messagelevel='error',
                                    caslib=model_weights_attr_caslib,
                                    path=model_weights_attr_path,
                                    casout=dict(replace=True, name=self.acoustic_model_weights_attr_name))
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Cannot load the acoustic model weights attribute.")

        rt = self.conn.retrieve("table.attribute",
                                _messagelevel='error',
                                name=self.acoustic_model_weights_name,
                                attrtable=self.acoustic_model_weights_attr_name,
                                task="ADD")
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Cannot add the weights attribute to the model weights.")

    def load_language_model(self, model_path, model_caslib=None):
        """
        Load the N-gram language model from the existing table.

        Parameters
        ----------
        model_path: string
            Specifies the path of the model table, relative to caslib_model.
        model_caslib : string, optional
            Specifies the caslib to load the model table.
            Notice that the current active caslib will be used if caslib_model is not specified.
            Default: None

        """
        if model_caslib is None:
            rt = self.conn.retrieve("langModel.lmImport",
                                    _messagelevel='error',
                                    table=dict(name=model_path),
                                    casout=dict(replace=True, name=self.language_model_name))
        else:
            rt = self.conn.retrieve("langModel.lmImport",
                                    _messagelevel='error',
                                    table=dict(caslib=model_caslib, name=model_path),
                                    casout=dict(replace=True, name=self.language_model_name))
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError("Cannot load the language model.")

    def transcribe(self, audio_path,
                   segment_len=30, gpu_devices=None, max_path_size=100, alpha=1.0, beta=0.0):
        """
        Transcribe the audio file into text.

        Notice that for this API, we are assuming that the speech-to-test models published by SAS Viya 3.4 will be used.
        Please download the acoustic and language model files from here:
        https://support.sas.com/documentation/prod-p/vdmml/zip/speech_19w21.zip

        Parameters
        ----------
        audio_path: string
            Specifies the location of the audio file (client-side, absolute/relative).

        segment_len: int, optional
            Specifies the maximum length of one segment in seconds.
            Default: 30
        gpu_devices: set of int, optional
            Specifies the gpu devices to use.
            Default: None
        max_path_size: int, optional
            Specifies the maximum number of paths kept as candidates of the final results during the decoding process.
            Default: 100
        alpha: double, optional
            Specifies the weight of the language model, relative to the acoustic model.
            Default: 1.0
        beta: double, optional
            Specifies the weight of the sentence length, relative to the acoustic model.
            Default: 0.0

        Returns
        -------
        result: string
        """

        listing_path_local, segment_path_list, segment_path_local_list = load_audio(
            self, audio_path, dict(name="audio", replace=True), segment_len=segment_len)
        extract_acoustic_features(self, "audio", dict(name="feature", replace=True))
        score_acoustic_features(self, "feature", dict(name="score", replace=True), gpu_devices=gpu_devices)
        decode_scores(self, "score", dict(name="result", replace=True),
                      max_path_size=max_path_size, alpha=alpha, beta=beta)
        result = concatenate_results(self, "result", segment_path_list)
        clean_audio(listing_path_local, segment_path_local_list)
        return result


def load_audio(speech, audio_path, casout, segment_len=30, framerate=16000, sampwidth=2):
    """
    Load the audio.

    Parameters
    ----------
    speech: class:"SpeechToText"
        Specifies the SpeechToText object.
    audio_path: string
        Specifies the location of the audio file (client-side, absolute/relative).
    casout: string, dict or CASTable
        Specifies the output CAS table.

    segment_len: int, optional
        Specifies the maximum length of one segment in seconds.
        Default: 30
    framerate: int, optional
        Specifies the desired sampling rate.
        Default: 16000
    sampwidth: int, optional
        Specifies the desired byte width.
        Default: 2

    Returns
    -------
    listing_path_local: string
        Return the location of the listing file (client-side).
    segment_path_list: list of string
        Return the location of the segmented audio files (server-side).
    segment_path_local_list: list of string
        Return the location of the segmented audio files (client-side).

    """
    listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list = \
        segment_audio(audio_path, speech.local_path, speech.data_path_after_caslib, segment_len, framerate, sampwidth)

    if speech.data_caslib is None:
        rt = speech.conn.retrieve("audio.loadAudio",
                                  _messagelevel='error',
                                  path=listing_path_after_caslib,
                                  casout=casout)
        caslib_path = speech.conn.caslibinfo(active=True).CASLibInfo["Path"][0]
    else:
        rt = speech.conn.retrieve("audio.loadAudio",
                                  _messagelevel='error',
                                  path=listing_path_after_caslib,
                                  caslib=speech.data_caslib,
                                  casout=casout)
        caslib_path = speech.conn.caslibinfo(caslib=speech.data_caslib).CASLibInfo["Path"][0]
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError("Cannot load the audio data.")

    if not caslib_path.endswith(speech.server_sep):
        caslib_path += speech.server_sep
    segment_path_list = [caslib_path + segment_path_after_caslib
                         for segment_path_after_caslib in segment_path_after_caslib_list]

    return listing_path_local, segment_path_list, segment_path_local_list


def extract_acoustic_features(speech, audio_table, casout,
                              frame_shift=10, frame_length=25, dither=0.0, n_bins=40, n_ceps=40,
                              feature_scaling_method="STANDARDIZATION", n_output_frames=3500):
    """
    Extract the audio features.

    Parameters
    ----------
    speech: class:"SpeechToText"
        Specifies the SpeechToText object.
    audio_table: string, dict or CASTable
        Specifies the CAS table containing the audio data.
    casout: string, dict or CASTable
        Specifies the output CAS table.

    frame_shift: int, optional
        Specifies the time difference (in milliseconds) between the beginnings of consecutive frames.
        Default: 10
    frame_length: int, optional
        Specifies the length of a frame (in milliseconds).
        Default: 25
    dither: double, optional
        Specifies the dithering constant (0.0 means no dithering).
        Default: 0.0
    n_bins: int, optional
        Specifies the number of triangular mel-frequency bins.
        Default: 40
    n_ceps: int, optional
        Specifies the number of cepstral coefficients in each MFCC feature frame.
        Default: 40
    feature_scaling_method: string, optional
        Specifies the feature scaling method to apply to the computed feature vectors.
        Default: "STANDARDIZATION"
    n_output_frames: int, optional
        Specifies the exact number of frames to include in the output table.
        Default: 3500
    """
    rt = speech.conn.retrieve("audio.computeFeatures",
                              _messagelevel='error',
                              table=audio_table,
                              casout=casout,
                              audioColumn="_audio_",
                              copyvars=["_path_"],
                              frameExtractionOptions=dict(frameShift=frame_shift,
                                                          frameLength=frame_length,
                                                          dither=dither),
                              melBanksOptions=dict(nbins=n_bins),
                              mfccOptions=dict(nceps=n_ceps),
                              featureScalingMethod=feature_scaling_method,
                              nOutputFrames=n_output_frames)
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError("Cannot extract the acoustic features.")


def score_acoustic_features(speech, feature_table, casout, gpu_devices=None):
    """
    Score the extracted features.

    Parameters
    ----------
    speech: class:"SpeechToText"
        Specifies the SpeechToText object.
    feature_table: string, dict or CASTable
        Specifies the CAS table containing the feature.
    casout: string, dict or CASTable
        Specifies the output CAS table.

    gpu_devices: set of int, optional
        Specifies the gpu devices to use.
        Default: None

    """
    if gpu_devices is not None:
        rt = speech.conn.retrieve("deepLearn.dlScore",
                                  _messagelevel='error',
                                  table=feature_table,
                                  casout=casout,
                                  model=speech.acoustic_model_name,
                                  initweights=speech.acoustic_model_weights_name,
                                  copyvars=["_path_"],
                                  gpu=dict(enable=True, devices=gpu_devices))
    else:
        rt = speech.conn.retrieve("deepLearn.dlScore",
                                  _messagelevel='error',
                                  table=feature_table,
                                  casout=casout,
                                  model=speech.acoustic_model_name,
                                  initweights=speech.acoustic_model_weights_name,
                                  copyvars=["_path_"])
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError("Cannot score the acoustic features.")


def decode_scores(speech, score_table, casout,
                  blank_label=" ", space_label="&", column_map=None, max_path_size=100, alpha=1.0, beta=0.0):
    """
    Decode the scores.

    Parameters
    ----------
    speech: class:"SpeechToText"
        Specifies the SpeechToText object.
    score_table: string, dict or CASTable
        Specifies the CAS table containing the score.
    casout: string, dict or CASTable
        Specifies the output CAS table.

    blank_label: string, optional
        Specifies the string used to indicate the blank label.
        Default: " "
    space_label: string, optional
        Specifies the string used to indicate the space label.
        Default: "&"
    column_map: list of string, optional
        Specifies the labels that the score columns in one time frame represent, which must follow the same order.
        Default: None
    max_path_size: int, optional
        Specifies the maximum number of paths kept as candidates of the final results during the decoding process.
        Default: 100
    alpha: double, optional
        Specifies the weight of the language model, relative to the acoustic model.
        Default: 1.0
    beta: double, optional
        Specifies the weight of the sentence length, relative to the acoustic model.
        Default: 0.0

    """
    if column_map is not None:
        rt = speech.conn.retrieve("langModel.lmDecode",
                                  _messagelevel='error',
                                  table=score_table,
                                  casout=casout,
                                  langModelTable=speech.language_model_name,
                                  blankLabel=blank_label,
                                  spaceLabel=space_label,
                                  column_map=column_map,
                                  maxPathSize=max_path_size,
                                  alpha=alpha,
                                  beta=beta,
                                  copyvars=["_path_"])
    else:
        rt = speech.conn.retrieve("langModel.lmDecode",
                                  _messagelevel='error',
                                  table=score_table,
                                  casout=casout,
                                  langModelTable=speech.language_model_name,
                                  blankLabel=blank_label,
                                  spaceLabel=space_label,
                                  maxPathSize=max_path_size,
                                  alpha=alpha,
                                  beta=beta,
                                  copyvars=["_path_"])
    if rt.severity > 1:
        for msg in rt.messages:
            print(msg)
        raise DLPyError("Cannot decode the scores.")


def concatenate_results(speech, result_table_name, segment_path_list):
    """
    Concatenate the results into a string.

    Parameters
    ----------
    speech: class:"SpeechToText"
        Specifies the SpeechToText object.
    result_table_name: string
        Specifies the CAS table containing the result.
    segment_path_list: list of string
        Specifies the locations of the segmented audio files (server-side).

    Returns
    -------
    result: string

    """
    result_table = speech.conn.CASTable(result_table_name)
    result_dict = dict(zip(list(result_table["_path_"]),
                           list(result_table["_audio_content_"])))
    result_list = [result_dict[segment_path] for segment_path in segment_path_list]
    result_list = [result.strip() for result in result_list]
    result_list = [result for result in result_list if len(result) > 0]
    return " ".join(result_list)

