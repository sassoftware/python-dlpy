from dlpy.utils import DLPyError, random_name
import wave
import audioop
import os
import math


def read_audio(path):
    """
    Read the audio from path into a wave_read object.

    Parameters
    ----------
    path: string
        Specifies the client-side path of the audio file.

    Returns
    -------
    wave_reader: class:'wave.Wave_read'
    wave_params: class:'wave._wave_params'

    """
    wave_reader = wave.open(path, "rb")
    wave_params = wave_reader.getparams()
    return wave_reader, wave_params


def check_framerate(params, framerate):
    """
    Check if the audio need to convert its frame rate.

    Parameters
    ----------
    params: class:'wave._wave_params'
        Specifies the original parameters of the audio file.
    framerate: int
        Specifies the desired frame rate.

    Returns
    -------
    boolean
        If the audio need to convert its frame rate.

    """
    return params.framerate == framerate


def check_sampwidth(params, sampwidth, sampwidth_options=(1, 2, 3, 4)):
    """
    Check if the audio need to convert its byte width.

    Parameters
    ----------
    params: class:'wave._wave_params'
        Specifies the original parameters of the audio file.
    sampwidth: int
        Specifies the desired byte width.
    sampwidth_options: tuple of int, optional
        Specifies the byte width options that we support.
        Default: (1, 2, 3, 4)

    Returns
    -------
    boolean
        If the audio needs to convert its byte width.

    """
    if params.sampwidth not in sampwidth_options:
        raise DLPyError("invalid wave file! Only byte width included in \"sampwidth_options\" "
                        "(default = (1, 2, 3, 4)) are accepted.")
    if sampwidth not in sampwidth_options:
        raise DLPyError("invalid desired byte width! Only byte width included in \"sampwidth_options\" "
                        "(default = (1, 2, 3, 4)) are accepted.")
    return params.sampwidth == sampwidth


def convert_framerate(fragment, width, nchannels, framerate_in, framerate_out):
    """
    Convert the frame rate of the input fragment.

    Parameters
    ----------
    fragment: bytes object
        Specifies the original fragment.
    width: int
        Specifies the fragment's original sample width in bytes.
    nchannels: int
        Specifies the fragment's original number of channels.
    framerate_in: int
        Specifies the fragment's original frame rate.
    framerate_out: int
        Specifies the fragment's desired frame rate.

    Returns
    -------
    new_fragment: bytes object

    """
    if framerate_in == framerate_out:
        return fragment

    new_fragment, _ = audioop.ratecv(fragment, width, nchannels, framerate_in, framerate_out, None)
    return new_fragment


def convert_sampwidth(fragment, sampwidth_in, sampwidth_out):
    """
    Convert the byte width of the input fragment.

    Parameters
    ----------
    fragment: bytes object
        Specifies the original fragment.
    sampwidth_in: int
        Specifies the fragment's original byte width.
    sampwidth_out: int
        Specifies the fragment's desired byte width.

    Returns
    -------
    new_fragment: bytes object

    """
    if sampwidth_in == sampwidth_out:
        return fragment

    # In .wav files, 16, 24, and 32 bit samples are signed, 8 bit samples are unsigned, so when converting
    # from 8 bit wide samples, you need to also subtract 128 from the sample.
    if sampwidth_in == 1:
        new_fragment = audioop.bias(fragment, 1, -128)
    else:
        new_fragment = fragment

    new_fragment = audioop.lin2lin(new_fragment, sampwidth_in, sampwidth_out)

    # When converting to 8 bit wide samples, you need to also add 128 to the result.
    if sampwidth_out == 1:
        new_fragment = audioop.bias(new_fragment, 1, 128)
    return new_fragment


def segment_audio(path, local_path, data_path_after_caslib, segment_len, framerate, sampwidth):
    """
    Segment the audio into shorter pieces and save the files.

    Parameters
    ----------
    path: string
        Specifies the client-side path of the audio file.
    local_path: string
        Specifies a location where temporary audio files are generated (client-side, absolute/relative).
        Notice that the path should be accessible by CAS server.
    data_path_after_caslib: string
        Specifies a location where temporary audio files are generated (server-side, relative to caslib_data).
        Notice that the path is relative to the caslib to use.
        Notice that data_path_after_caslib and local_path point to the same location.
    segment_len: int
        Specifies the maximum length of one segment in seconds.
    framerate: int
        Specifies the desired sampling rate.
    sampwidth: int
        Specifies the desired byte width.

    Returns
    -------
    listing_path_after_caslib: string
    listing_path_local: string
    segment_path_after_caslib_list: list of string
    segment_path_local_list: list of string

    """

    if os.path.isfile(path):
        wave_reader, wave_params = read_audio(path)
    else:
        raise DLPyError("Cannot find the audio file.")

    is_framerate_correct = check_framerate(wave_params, framerate)
    is_sampwidth_correct = check_sampwidth(wave_params, sampwidth)

    # calculate the number of segments to split
    nframes_of_total = wave_params.nframes
    nframes_of_segment = segment_len * wave_params.framerate
    num_of_segments = math.ceil(1.0 * nframes_of_total / nframes_of_segment)

    # generate the listing file name
    audio_name = os.path.basename(path)
    audio_name = os.path.splitext(audio_name)[0]
    listing_name = None
    while not listing_name:
        listing_name = random_name(audio_name, 6) + ".listing"
        listing_path_after_caslib = data_path_after_caslib + listing_name
        listing_path_local = os.path.join(local_path, listing_name)
        if os.path.exists(listing_path_local):
            listing_name = None

    # segmentation
    segment_path_after_caslib_list = []
    segment_path_local_list = []
    with open(listing_path_local, "w") as listing_file:
        wave_reader.rewind()
        for i in range(num_of_segments):
            segment_name = None
            segment_path_after_caslib = None
            segment_path_local = None
            while not segment_name:
                segment_name = random_name(audio_name + str(i), 6) + ".wav"
                segment_path_after_caslib = data_path_after_caslib + segment_name
                segment_path_local = os.path.join(local_path, segment_name)
                if os.path.exists(segment_path_local):
                    segment_name = None
            with wave.open(segment_path_local, "wb") as wave_writer:
                segment_path_after_caslib_list.append(segment_path_after_caslib)
                segment_path_local_list.append(segment_path_local)
                wave_writer.setnchannels(wave_params.nchannels)
                wave_writer.setframerate(framerate)
                wave_writer.setsampwidth(sampwidth)
                wave_writer.setcomptype(wave_params.comptype, wave_params.compname)
                fragment = wave_reader.readframes(nframes_of_segment)
                if not is_framerate_correct:
                    fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                                 wave_params.framerate, framerate)
                if not is_sampwidth_correct:
                    fragment = convert_sampwidth(fragment, wave_params.sampwidth, sampwidth)
                wave_writer.writeframes(fragment)
        wave_reader.close()

        for segment_path_after_caslib in segment_path_after_caslib_list:
            listing_file.write(segment_path_after_caslib + "\n")

    return listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list


def clean_audio(listing_path_local, segment_path_local_list):
    """
    Remove the temporary listing file and the temporary audio files generated.

    Parameters
    ----------
    listing_path_local: string
        Specifies the location of the temporary listing file generated (client-side, absolute/relative).
    segment_path_local_list: list of string
        Specifies the locations of the temporary audio files generated (client-side, absolute/relative).

    """
    if os.path.exists(listing_path_local):
        os.remove(listing_path_local)
    for segment_path_local in segment_path_local_list:
        if os.path.exists(segment_path_local):
            os.remove(segment_path_local)
