from dlpy.utils import DLPyError, random_name
import wave
import audioop
import os


def read_audio(path):
    """
    Read the audio from path into a wave_read object.

    Parameters
    ----------
    path : string
        Specifies path of the audio file.

    Returns
    -------
    wave_reader : class : 'wave.Wave_read'
    wave_params : class : 'wave._wave_params'

    """
    wave_reader = wave.open(path, "rb")
    wave_params = wave_reader.getparams()
    return wave_reader, wave_params


def check_framerate(params, framerate):
    """
    Check if the input audio has the desired framerate (sampling rate).

    Parameters
    ----------
    params : class : 'wave._wave_params'
        Specifies the original parameters of the audio.
    framerate : int
        Specifies the desired framerate.

    Returns
    -------
    boolean

    """
    return params.framerate == framerate


def check_sampwidth(params, sampwidth):
    """
    Check if the input audio has the desired sampwdith (byte width).

    Parameters
    ----------
    params : class : 'wave._wave_params'
        Specifies the original parameters of the audio.
    sampwidth : int
        Specifies the desired sampwidth.

    Returns
    -------
    boolean

    """
    if params.sampwidth not in {1, 2, 3, 4}:
        raise DLPyError("invalid wave input! Only byte width values included in {1, 2, 3, 4} are accepted.")
    if sampwidth not in {1, 2, 3, 4}:
        raise DLPyError("invalid desired byte width! Only byte width values included in {1, 2, 3, 4} are accepted.")
    return params.sampwidth == sampwidth


def convert_framerate(fragment, width, nchannels, framerate_in, framerate_out):
    """
    Convert framerate (sampling rate) of the input fragment.

    Parameters
    ----------
    fragment : bytes object
        Specifies the original fragment.
    width : int
        Specifies the fragment's original sampwidth.
    nchannels : int
        Specifies the fragment's original nchannels.
    framerate_in : int
        Specifies the fragment's original framerate.
    framerate_out : int
        Specifies the fragment's desired framerate.

    Returns
    -------
    bytes

    """
    if framerate_in == framerate_out:
        return fragment

    new_fragment, _ = audioop.ratecv(fragment, width, nchannels, framerate_in, framerate_out, None)
    return new_fragment


def convert_sampwidth(fragment, sampwidth_in, sampwidth_out):
    """
    Convert the sampwidth (byte width) of the input fragment between 1-, 2-, 3-, 4-byte formats.

    Parameters
    ----------
    fragment : bytes object
        Specifies the original fragment.
    sampwidth_in : int
        Specifies the fragment's original sampwidth.
    sampwidth_out : int
        Specifies the fragment's desired sampwidth.

    Returns
    -------
    bytes

    """
    if sampwidth_in == sampwidth_out:
        return fragment

    # In .wav files, 16, 24, and 32 bit samples are signed, 8 bit samples are unsigned.
    # So when converting from 8 bit wide samples, you need to also subtract 128 from the sample.
    # Similarly, when converting to 8 bit wide samples, you need to also add 128 to the result.
    if sampwidth_in == 1:
        new_fragment = audioop.bias(fragment, 1, -128)
    else:
        new_fragment = fragment
    new_fragment = audioop.lin2lin(new_fragment, sampwidth_in, sampwidth_out)
    if sampwidth_out == 1:
        new_fragment = audioop.bias(new_fragment, 1, 128)
    return new_fragment


def calculate_segment_nframes(path, segment_len):
    """
    Calculate the number of frames of every segment split from the audio input.

    Parameters
    ----------
    path : string
        Specifies path of the audio file.
    segment_len : float
        Specifies the maximum length of one segment in seconds.

    Returns
    -------
    list of ints
    """

    wave_reader, wave_params = read_audio(path)
    window_nframes = int(wave_params.framerate * 0.01)  # every window last 0.01 second
    segment_nframes = int(wave_params.framerate * segment_len)

    # switch every window by 0.01 second
    # save the frame index of middle of the window to frame_list
    # save maximum value of the window to max_list
    frame = 0
    frame_list, max_list = [], []
    while True:
        if frame >= wave_params.nframes:
            break
        fragment = wave_reader.readframes(window_nframes)
        frame_list.append(min(int(frame + window_nframes / 2),
                              wave_params.nframes))
        max_list.append(audioop.max(fragment, wave_params.sampwidth))
        frame += window_nframes
    wave_reader.close()

    # calculate the threshold by 30 percentile
    max_list_sorted = sorted(max_list)
    threshold = max_list_sorted[int(len(max_list_sorted) * 30. / 100)]

    # calculate how many previous windows have maximum values smaller than threshold
    continuous = 0
    continuous_list = []
    for max_val in max_list:
        if max_val < threshold:
            continuous += 1
        else:
            continuous = 0
        continuous_list.append(continuous)

    # find frame numbers of breakpoints
    breakpoint_frame_list = []
    while True:
        frame_min = frame_list[0]
        frame_max = frame_min + segment_nframes - window_nframes
        if frame_list[-1] <= frame_max:
            break

        for index, frame in enumerate(frame_list):
            if frame > frame_max:
                continuous_max_value = max(continuous_list[:index])
                continuous_max_index = continuous_list.index(continuous_max_value)
                for i in range(continuous_max_index + 1):
                    continuous_list[i] = 0

                continuous_max_index = int(continuous_max_index - (continuous_max_value - 1) / 2)
                breakpoint_frame_list.append(frame_list[continuous_max_index])
                frame_list = frame_list[continuous_max_index + 1:]
                continuous_list = continuous_list[continuous_max_index + 1:]
                break

    # remove too close breakpoints
    i = 1
    while True:
        if len(breakpoint_frame_list) < 2 or i >= len(breakpoint_frame_list):
            break
        if i == 1:
            if breakpoint_frame_list[i] < segment_nframes:
                del breakpoint_frame_list[0]
            else:
                i += 1
        else:
            if breakpoint_frame_list[i] - breakpoint_frame_list[i - 2] < segment_nframes:
                del breakpoint_frame_list[i - 1]
            else:
                i += 1

    # calculate nframes_list
    segment_nframes_list = []
    if len(breakpoint_frame_list) > 0:
        segment_nframes_list.append(breakpoint_frame_list[0])
    for i in range(1, len(breakpoint_frame_list)):
        segment_nframes_list.append(breakpoint_frame_list[i] - breakpoint_frame_list[i - 1])
    if len(breakpoint_frame_list) == 0 or breakpoint_frame_list[-1] < wave_params.nframes:
        segment_nframes_list.append(segment_nframes)
    return segment_nframes_list


def segment_audio(path, local_path, data_path_after_caslib, segment_len, framerate, sampwidth):
    """
    Segment the audio into pieces shorter than segment_len.

    Parameters
    ----------
    path : string
        Specifies path of the audio file.
    local_path : string
        Specifies the location where temporary segmented audio files are stored (server side).
    data_path_after_caslib : string
        Specifies the location where temporary segmented audio files are stored (client side, relative to caslib).
        Note that local_path and data_path_after_caslib actually point to the same position.
    segment_len : float
        Specifies the maximum length of one segment in seconds.
    framerate : int
        Specifies the desired framerate.
    sampwidth : int
        Specifies the desired sampwidth.

    Returns
    -------
    listing_path_after_caslib : string
    listing_path_local : string
    segment_path_after_caslib_list : list of string
    segment_path_local_list : list of string

    """

    if os.path.isfile(path):
        wave_reader, wave_params = read_audio(path)
    else:
        raise DLPyError("Cannot find the audio file.")

    if segment_len <= 0:
        raise DLPyError("Incorrect \"segment_len\" value: the segment length maximum can only be positive.")
    if segment_len > 35:
        raise DLPyError("Incorrect \"segment_len\" value: the segment length maximum cannot be longer than 35 seconds.")

    is_framerate_desired = check_framerate(wave_params, framerate)
    is_sampwidth_desired = check_sampwidth(wave_params, sampwidth)

    # generate the listing file name
    audio_name = os.path.basename(path)
    audio_name = os.path.splitext(audio_name)[0]
    listing_name_no_ext = None
    listing_name = None
    while listing_name is None:
        listing_name_no_ext = random_name(audio_name, 6)
        listing_name = listing_name_no_ext + ".listing"
        listing_path_after_caslib = data_path_after_caslib + listing_name
        listing_path_local = os.path.join(local_path, listing_name)
        if os.path.exists(listing_path_local):
            listing_name = None

    # segmentation
    segment_nframes_list = calculate_segment_nframes(path, segment_len)
    print("Note:", str(len(segment_nframes_list)), "temporary audio files are created.")

    segment_path_after_caslib_list = []
    segment_path_local_list = []
    with open(listing_path_local, "w") as listing_file:
        wave_reader.rewind()
        for i in range(len(segment_nframes_list)):
            segment_name = listing_name_no_ext + "_" + str(i) + ".wav"
            segment_path_after_caslib = data_path_after_caslib + segment_name
            segment_path_local = os.path.join(local_path, segment_name)

            with wave.open(segment_path_local, "wb") as wave_writer:
                segment_path_after_caslib_list.append(segment_path_after_caslib)
                segment_path_local_list.append(segment_path_local)
                wave_writer.setnchannels(wave_params.nchannels)
                wave_writer.setframerate(framerate)
                wave_writer.setsampwidth(sampwidth)
                wave_writer.setcomptype(wave_params.comptype, wave_params.compname)
                fragment = wave_reader.readframes(segment_nframes_list[i])
                if not is_framerate_desired:
                    fragment = convert_framerate(fragment, wave_params.sampwidth, wave_params.nchannels,
                                                 wave_params.framerate, framerate)
                if not is_sampwidth_desired:
                    fragment = convert_sampwidth(fragment, wave_params.sampwidth, sampwidth)
                wave_writer.writeframes(fragment)
        wave_reader.close()

        for segment_path_after_caslib in segment_path_after_caslib_list:
            listing_file.write(segment_path_after_caslib + "\n")

    # listing_path_after_caslib: to load audio
    # listing_path_local: to remove listing file
    # segment_path_after_caslib_list: to concatenate results (add caslib path)
    # segment_path_local_list: to remove segmented files
    return listing_path_after_caslib, listing_path_local, segment_path_after_caslib_list, segment_path_local_list


def clean_audio(listing_path_local, segment_path_local_list):
    """
    Remove the temporary listing file and the temporary audio files.

    Parameters
    ----------
    listing_path_local : string
        Specifies path of the temporary listing file to remove.
    segment_path_local_list : list of string
        Specifies paths of the temporary audio files to remove.

    """
    if os.path.exists(listing_path_local):
        os.remove(listing_path_local)
    for segment_path_local in segment_path_local_list:
        if os.path.exists(segment_path_local):
            os.remove(segment_path_local)
    print("Note: all temporary files are removed.")
