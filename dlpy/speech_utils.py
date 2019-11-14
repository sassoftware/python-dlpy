import random

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
        'wave.Wave_read' Object returned by opening the audio file listed in 'path'.
    wave_params : class : 'wave._wave_params'
        Wave parameters (nchannels, sampwidth, framerate, nframes, comptype, compname) obtained by calling getparams()
        on the 'wave.Wave_read' Object.

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
    Boolean
        Whether the input audio has the desired framerate (True) or not (False).

    """
    return params.framerate == framerate


def check_sampwidth(params, sampwidth):
    """
    Check if the input audio has the desired sampwidth (byte width).

    Parameters
    ----------
    params : class : 'wave._wave_params'
        Specifies the original parameters of the audio.
    sampwidth : int
        Specifies the desired sampwidth.

    Returns
    -------
    Boolean
        Whether the input audio has the desired sampwidth (True) or not (False).

    """
    if params.sampwidth not in {1, 2, 3, 4}:
        raise DLPyError("invalid wave input! Only byte width values included in {1, 2, 3, 4} are accepted.")
    if sampwidth not in {1, 2, 3, 4}:
        raise DLPyError("invalid desired byte width! Only byte width values included in {1, 2, 3, 4} are accepted.")
    return params.sampwidth == sampwidth


def check_stereo(params):
    """
    Check if the input audio has 2 channels (stereo).

    Parameters
    ----------
    params : class : 'wave._wave_params'
        Specifies the original parameters of the audio.

    Returns
    -------
    Boolean
        Whether the input audio has 2 channels (True) or not (False).

    """
    if params.nchannels not in {1, 2}:
        raise DLPyError("invalid wave input! Only mono and stereo are supported.")
    return params.nchannels == 2


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
        Converted audio with the desired framerate 'framerate_out'.

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
        Converted audio with the desired sampwidth 'sampwidth_out'.

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


def convert_stereo_to_mono(fragment, width):
    """
    Convert stereo fragment to mono.

    Parameters
    ----------
    fragment : bytes object
        Specifies the original fragment.
    width : int
        Specifies the fragment's original sampwidth.

    Returns
    -------
    bytes
        Converted audio in mono type.

    """
    new_fragment = audioop.tomono(fragment, width, 0.5, 0.5)
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
        A list of each segment length in frames.
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
        Path of the file listing the audio segments on the server side, relative to caslib.
    listing_path_local : string
        Path of the file listing the audio segments on the client side.
    segment_path_after_caslib_list : list of string
        A list of paths of the audio segments on the server side, relative to caslib.
    segment_path_local_list : list of string
        A list of paths of the audio segments on client side.

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
    is_stereo = check_stereo(wave_params)

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
                wave_writer.setnchannels(1)
                wave_writer.setframerate(framerate)
                wave_writer.setsampwidth(sampwidth)
                wave_writer.setcomptype(wave_params.comptype, wave_params.compname)
                fragment = wave_reader.readframes(segment_nframes_list[i])
                if is_stereo:
                    fragment = convert_stereo_to_mono(fragment, wave_params.sampwidth)

                if not is_framerate_desired:
                    fragment = convert_framerate(fragment, wave_params.sampwidth, 1,
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
    is_removed = False
    if os.path.exists(listing_path_local):
        os.remove(listing_path_local)
        is_removed = True
    for segment_path_local in segment_path_local_list:
        if os.path.exists(segment_path_local):
            os.remove(segment_path_local)
            is_removed = True
    if is_removed:
        print("Note: all temporary files are removed.")


def random_file_from_dir(local_audio_file):
    n=0
    random.seed()
    for r, d, f in os.walk(local_audio_file):
      for name in f:
        n=n+1
        if random.uniform(0, n) < 1:
            random_file=os.path.join(r, name)
    return random_file


def play_one_audio_file(local_audio_file):
    '''
    Play a local audio file using soundfile and sounddevice.

    Parameters
    ----------
    local_audio_file : string
        Local location to the audio file to be played. When it is a directory,
        a file will be randomly chosen.

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    try:
        import soundfile as sf
        import sounddevice as sd
    except (ModuleNotFoundError, ImportError):
        raise DLPyError('cannot import soundfile or sounddevice')

    if os.path.isdir(local_audio_file):
        local_audio_file_real = random_file_from_dir(local_audio_file)
    else:
        local_audio_file_real = local_audio_file

    print('File location: {}'.format(local_audio_file_real))

    data, sampling_rate = sf.read(local_audio_file_real)

    print('Frequency [Hz]: {}'.format(sampling_rate))
    print('Duration [s]: {}'.format(data.shape[0]/sampling_rate))
    sd.play(data, sampling_rate)
    sd.wait()

def display_spectrogram_for_one_audio_file(local_audio_file):
    '''
    Display spectrogram for a local audio file using soundfile.

    Parameters
    ----------
    local_audio_file : string
        Local location to the audio file to be displayed.

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    try:
        import soundfile as sf
        import matplotlib.pylab as plt
    except (ModuleNotFoundError, ImportError):
        raise DLPyError('cannot import soundfile')

    if os.path.isdir(local_audio_file):
        local_audio_file_real = random_file_from_dir(local_audio_file)
    else:
        local_audio_file_real = local_audio_file

    print('File location: {}'.format(local_audio_file_real))

    data, sampling_rate = sf.read(local_audio_file_real)

    plt.specgram(data, Fs=sampling_rate)
    # add axis labels
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')


def display_raw_data_for_one_audio_file(local_audio_file):
    '''
    Display raw data for a local audio file using soundfile.

    Parameters
    ----------
    local_audio_file : string
        Local location to the audio file to be displayed.

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    try:
        import soundfile as sf
        import matplotlib.pylab as plt
    except (ModuleNotFoundError, ImportError):
        raise DLPyError('cannot import soundfile')

    if os.path.isdir(local_audio_file):
        local_audio_file_real = random_file_from_dir(local_audio_file)
    else:
        local_audio_file_real = local_audio_file

    print('File location: {}'.format(local_audio_file_real))

    data, sampling_rate = sf.read(local_audio_file_real)

    plt.plot(data)


def convert_one_audio_file(local_audio_file, converted_local_audio_file):
    '''
    Convert a local audio file into a wav format that only contains 1 channel with 16 bits and 16K HZ.

    Parameters
    ----------
    local_audio_file : string
        Local location to the audio file to be converted.

    converted_local_audio_file : string
        Local location to store the converted audio file

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    try:
        import soundfile as sf
    except (ModuleNotFoundError, ImportError):
        raise DLPyError('cannot import soundfile')

    audio_name = os.path.basename(local_audio_file)
    output_dir = os.path.dirname(converted_local_audio_file)
    required_sr = 16000
    required_sw = 2

    # check whether the audio file is a wave format
    audio_ext = os.path.splitext(audio_name)[-1]
    audio_name = os.path.splitext(audio_name)[0]
    if audio_ext.lower() != '.wav':
        audio_wav_file = output_dir + random_name(audio_name, 6) + '.wav'
        data, sampling_rate = sf.read(local_audio_file)
        sf.write(audio_wav_file, data, sampling_rate)
    else:
        audio_wav_file = local_audio_file

    # convert the wav file to the required format: 1 channel, 16 bits, and 16K HZ
    wave_reader, wave_params = read_audio(audio_wav_file)
    is_framerate_desired = check_framerate(wave_params, required_sr)
    is_sampwidth_desired = check_sampwidth(wave_params, required_sw)
    is_stereo = check_stereo(wave_params)

    if converted_local_audio_file == audio_wav_file:
        real_converted_local_audio_file = converted_local_audio_file + '.tmp'
    else:
        real_converted_local_audio_file = converted_local_audio_file

    with wave.open(real_converted_local_audio_file, "wb") as wave_writer:
        wave_writer.setnchannels(1)
        wave_writer.setframerate(required_sr)
        # 16 bits
        wave_writer.setsampwidth(2)
        wave_writer.setcomptype(wave_params.comptype, wave_params.compname)
        fragment = wave_reader.readframes(wave_params.nframes)

        # 1 channel
        if is_stereo:
            fragment = convert_stereo_to_mono(fragment, wave_params.sampwidth)

        # 16K HZ
        if not is_framerate_desired:
            fragment = convert_framerate(fragment, wave_params.sampwidth, 1,
                                         wave_params.framerate, required_sr)

        # 16 bits
        if not is_sampwidth_desired:
            fragment = convert_sampwidth(fragment, wave_params.sampwidth, required_sw)

        wave_writer.writeframes(fragment)

    wave_reader.close()

    # remove the temporary wav file
    if audio_wav_file != local_audio_file:
        os.remove(audio_wav_file)

    # rename the file to the desired one
    if real_converted_local_audio_file != converted_local_audio_file:
        os.replace(real_converted_local_audio_file, converted_local_audio_file)


def convert_audio_files(local_audio_path, recurse=True):
    '''
    Convert audio files under a local path into wave files that only contains 1 channel with 16 bits and 16K HZ.

    Parameters
    ----------
    local_audio_path : string
        Local location to the audio files that will be converted. The new wave files will be stored under this path.
        Note if the files are already in the wave format, they will be overwritten.

    recurse : bool, optional
        Specifies whether to recursively convert all the audio files.
        Default : True

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    number_files = 0

    if recurse:
        for r, d, f in os.walk(local_audio_path):
            number_files = number_files + len(f)
    else:
        for f in os.listdir(local_audio_path):
            local_file = os.path.join(local_audio_path, f)
            if os.path.isfile(local_file):
                number_files = number_files + 1

    print('File path: {}'.format(local_audio_path))
    print('Number of Files: {}'.format(number_files))

    print_freq = 1000

    number_files = 0
    if recurse:
        for r, d, f in os.walk(local_audio_path):
            for file in f:
                local_file = os.path.join(r, file)
                local_file_wav = os.path.splitext(local_file)[0] + '.wav'
                try:
                    convert_one_audio_file(local_file, local_file_wav)
                    number_files = number_files + 1
                except:
                    print('Cannot convert file {}'.format(local_file))

                if number_files % print_freq == 0:
                    print('Number of files processed: {}'.format(number_files))
    else:
        for f in os.listdir(local_audio_path):
            local_file = os.path.join(local_audio_path, f)
            if os.path.isfile(local_file):
                local_file_wav = os.path.join(local_audio_path, os.path.splitext(f)[0] + '.wav')
                try:
                    convert_one_audio_file(local_file, local_file_wav)
                    number_files = number_files + 1
                except:
                    print('Cannot convert file {}'.format(local_file))

                if number_files % print_freq == 0:
                    print('Number of files processed: {}'.format(number_files))

    print('File conversions are finished.')


def convert_one_audio_file_to_specgram(local_audio_file, converted_local_png_file):
    '''
    Convert a local audio file into a png format with spectrogram.

    Parameters
    ----------
    local_audio_file : string
        Local location to the audio file to be converted.

    converted_local_png_file : string
        Local location to store the converted audio file

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    try:
        import soundfile as sf
        import matplotlib.pylab as plt
    except (ModuleNotFoundError, ImportError):
        raise DLPyError('cannot import soundfile')

    data, sampling_rate = sf.read(local_audio_file)

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    ax.specgram(x=data, Fs=sampling_rate)
    ax.axis('off')
    fig.savefig(converted_local_png_file, dpi=300, frameon='false')
    # this is the key to avoid mem leaking in notebook
    plt.ioff()
    plt.close(fig)


def convert_audio_files_to_specgrams(local_audio_path, recurse=True):
    '''
    Convert audio files under a local path into the images (PNG) that contain spectrogram.

    Parameters
    ----------
    local_audio_path : string
        Local location to the audio files that will be converted. The new image files will be stored under this path.
        Note if the files are already in the PNG format, they will be overwritten.

    recurse : bool, optional
        Specifies whether to recursively convert all the audio files.
        Default : True

    Returns
    -------
    None

    Raises
    ------
    DLPyError
        If anything goes wrong, it complains and prints the appropriate message.

    '''

    number_files = 0

    if recurse:
        for r, d, f in os.walk(local_audio_path):
            number_files = number_files + len(f)
    else:
        for f in os.listdir(local_audio_path):
            local_file = os.path.join(local_audio_path, f)
            if os.path.isfile(local_file):
                number_files = number_files + 1

    print('File path: {}'.format(local_audio_path))
    print('Number of Files: {}'.format(number_files))

    print_freq = 1000

    number_files = 0
    if recurse:
        for r, d, f in os.walk(local_audio_path):
            for file in f:
                local_file = os.path.join(r, file)
                local_file_png = os.path.splitext(local_file)[0] + '.png'
                try:
                    convert_one_audio_file_to_specgram(local_file, local_file_png)
                    number_files = number_files + 1
                except:
                    print('Cannot convert file {}'.format(local_file))
                if number_files % print_freq == 0:
                    print('Number of files processed: {}'.format(number_files))
    else:
        for f in os.listdir(local_audio_path):
            local_file = os.path.join(local_audio_path, f)
            if os.path.isfile(local_file):
                local_file_png = os.path.join(local_audio_path, os.path.splitext(f)[0] + '.png')
                try:
                    convert_one_audio_file_to_specgram(local_file, local_file_png)
                    number_files = number_files + 1
                except:
                    print('Cannot convert file {}'.format(local_file))
                if number_files % print_freq == 0:
                    print('Number of files processed: {}'.format(number_files))

    print('File conversions are finished.')

