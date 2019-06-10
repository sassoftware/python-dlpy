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

''' Audio related classes and functions '''

from swat.cas.table import CASTable
from .data_clean import DataClean
from .utils import random_name, find_caslib, get_cas_host_type, find_path_of_caslib
from dlpy.utils import DLPyError


class AudioTable(CASTable):
    ''' CAS Table for Audio '''

    running_caslib = None

    @classmethod
    def load_audio_files(cls, conn, path, casout=None, caslib=None):
        '''
        Load audio files from path

        Parameters
        ----------
        conn : CAS
            CAS connection object
        path : string
            Path to audio files
        casout : dict, string, or CASTable, optional
            The output CAS table specification
        caslib : string, optional
            The caslib to load audio files from

        Returns
        -------
        :class:`AudioTable`
            If audio files are found
        None
            If no audio files are found

        '''
        conn.loadactionset('audio', _messagelevel='error')

        if casout is None:
            casout = dict(name=random_name('AudioTable', 6))
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if caslib is None:
            # get the os of the server
            server_type = get_cas_host_type(conn).lower()
            if server_type.startswith("lin") or server_type.startswith("osx"):
                path_split = path.rsplit("/", 1)
            else:
                path_split = path.rsplit("\\", 1)

            # try accessing the file
            if len(path_split) == 2:
                caslib = find_caslib(conn, path_split[0])
                if caslib is not None:
                    rt2 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                        caslib=caslib, path=path_split[1])
                    if rt2.severity > 1:
                        for msg in rt2.messages:
                            print(msg)
                        raise DLPyError('cannot load audio files, something is wrong!')
                    cls.running_caslib = path_split[0]
                    return AudioTable(casout['name'])
                else:
                    caslib = random_name('Caslib', 6)
                    rt2 = conn.retrieve('addcaslib', _messagelevel='error', name=caslib, path=path_split[0],
                                        activeonadd=False, subdirectories=True, datasource={'srctype':'path'})
                    if rt2.severity < 2:
                        rt3 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                            caslib=caslib, path=path_split[1])
                        if rt3.severity > 1:
                            for msg in rt3.messages:
                                print(msg)
                            raise DLPyError('cannot load audio files, something is wrong!')
                        else:
                            cls.running_caslib = path_split[0]
                            return AudioTable(casout['name'])
            return None
        else:
            rt4 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                caslib=caslib, path=path)
            if rt4.severity > 1:
                for msg in rt4.messages:
                    print(msg)
                raise DLPyError('cannot load audio files, something is wrong!')
            cls.running_caslib = find_path_of_caslib(conn, caslib)
            return AudioTable(casout['name'])

    @classmethod
    def extract_audio_features(cls, conn, table, frame_shift=10, frame_length=25, n_bins=40, n_ceps=40,
                               feature_scaling_method='STANDARDIZATION', n_output_frames=500, casout=None, **kwargs):
        '''
        Extracts audio features from the audio files

        Parameters
        ----------
        conn : CAS
            A connection object to the current session.
        table : AudioTable
            An audio table containing the audio files.
        frame_shift : int, optional
            Specifies the time difference (in milliseconds) between the beginnings of consecutive frames.
            Default: 10
        frame_length : int, optional
            Specifies the length of a frame (in milliseconds).
            Default: 25
        n_bins : int, optional
            Specifies the number of triangular mel-frequency bins.
            Default: 40
        n_ceps : int, optional
            Specifies the number of cepstral coefficients in each MFCC feature frame (including C0).
            Default: 40
        feature_scaling_method : string, optional
            Specifies the feature scaling method to apply to the computed feature vectors.
            Default: 'standardization'
        n_output_frames : int, optional
            Specifies the exact number of frames to include in the output table (extra frames are dropped and missing
            frames are padded with zeros).
            Default: 500
        casout : dict or string or CASTable, optional
            CAS Output table
        kwargs : keyword-arguments, optional
            Additional parameter for feature extraction.

        Returns
        -------
        :class:`AudioTable`
            If table exists
        None
            If no table exists

        '''
        if isinstance(table, AudioTable) or isinstance(table, CASTable):
            if casout is None:
                casout = dict(name=random_name('AudioTable', 6))
            elif isinstance(casout, CASTable) or isinstance(casout, AudioTable):
                casout = casout.to_outtable_params()

            rt = conn.retrieve('audio.computefeatures', _messagelevel='error', table=table,
                               frameExtractionOptions=dict(frameshift=frame_shift, framelength=frame_length),
                               melBanksOptions=dict(nbins=n_bins), mfccOptions=dict(nceps=n_ceps),
                               featureScalingMethod=feature_scaling_method, nOutputFrames=n_output_frames,
                               casout=casout, **kwargs)
            if rt.severity > 1:
                for msg in rt.messages:
                    print(msg)
                return None

            return AudioTable(casout['name'])

        return None

    @classmethod
    def load_audio_metadata_speechrecognition(cls, conn, path, audio_path):
        '''
        Pre-process and loads the metadata

        Parameters
        ----------
        conn : CAS
            A connection object to the current session.
        path : string
            Location to the input metadata file.
        audio_path : delimiter
            Delimiter for the metadata file.

        Returns
        -------
        :class:`CASTable`

        '''
        output_name = random_name('AudioTable_Metadata', 6)
        
        dc = DataClean(conn=conn, contents_as_path=path)
        dc_response = dc.process_contents(audio_path = audio_path)
        tbl = dc.create_castable(dc_response['results'], output_name, replace=True, promote=False,
                                 col_names=dc_response['col_names'])
     
        return tbl

    @classmethod
    def load_audio_metadata(cls, conn, path, audio_path, task='speech2text'):
        '''
        Pre-process and loads the metadata

        Parameters
        ----------
        conn : CAS
            A connection object to the current session.
        path : string
            Location to the input metadata file.
        audio_path : string
            Location to the audio files.
        task : string, optional
            Specifies the task
            Note: currently only support 'speech2text' (default)

        Returns
        -------
        :class:`CASTable`

        Raises
        ------
        DLPyError
            If anything goes wrong, it complains and prints the appropriate message.

        '''
        
        if task == 'speech2text':
            return cls.load_audio_metadata_speechrecognition(conn, path, audio_path)
        else:
            raise DLPyError("We do not support this task yet!")

    @classmethod
    def create_audio_table(cls, conn, data_path, metadata_path,
                           features_parameters=dict(frame_shift=10, frame_length=25, n_bins=40, n_ceps=40,
                                                    feature_scaling_method='STANDARDIZATION', n_output_frames=500),
                           casout=None,
                           task='speech2text'):
        '''
        Creates an Audio table and takes care of all the necessary steps

        Parameters
        ----------
        conn : CAS
            A connection object to the current session.
        data_path : string
            Path to the file that contains the list of audio files (this is
            expected to be on the server side).
        metadata_path : string
            Location to the metadata file (this is expected to be on the client side).
        features_parameters : dict, optional
            Parameters to be used while extracting audio features
        casout : string, dict, or CASTable, optional
            Resulting output CAS table
        task : string, optional
            Specifies the type of the task. Default is speech to text.
            Note: currently only support 'speech2text' (default)

        Returns
        -------
        :class:`AudioTable`
            A table containing audio features of audio files as well as their labels.
            The resulting table can be directly used in the deep learning models.

        Raises
        ------
        DLPyError
            If anything goes wrong at any point in the process of creating this AudioTable, it complains and
            prints the appropriate message.

        '''

        if task == 'speech2text':
            return cls.create_audio_table_speechrecognition(conn, data_path, metadata_path,
                                                            features_parameters=features_parameters,
                                                            casout=casout)
        else:
            raise DLPyError("We do not support this task!")


    @classmethod
    def create_audio_table_speechrecognition(cls, conn, data_path, metadata_path,
                                             features_parameters=dict(frame_shift=10, frame_length=25, n_bins=40,
                                                                      n_ceps=40,
                                                                      feature_scaling_method='STANDARDIZATION',
                                                                      n_output_frames=500),
                                             casout=None):
        '''
        Creates an Audio table and takes care of all the necessary steps

        Parameters
        ----------
        conn : CAS
            A connection object to the current session.
        data_path : string
            Path to the file that contains the list of audio files (this is
            expected to be on the server side).
        metadata_path : string
            Location to the metadata file (this is expected to be on the client side).
        features_parameters : dict, optional
            Parameters to be used while extracting audio features
        casout : string, optional
            Resulting output CAS table

        Returns
        -------
        :class:`AudioTable`
            A table containing audio features of audio files as well as their labels. The resulting table can be
            directly used in the deep learning models.

        Raises
        ------
        DLPyError
            If anything goes wrong at any point in the process of creating this AudioTable, it complains and
            prints the appropriate message.

        '''
        au = cls.load_audio_files(conn, data_path)
        if au is None:
            raise DLPyError('cannot load audio files')

        fp = features_parameters

        features = cls.extract_audio_features(conn, au, frame_shift=fp['frame_shift'], frame_length=fp['frame_length'],
                                              n_bins=fp['n_bins'], n_ceps=fp['n_ceps'],
                                              feature_scaling_method=fp['feature_scaling_method'],
                                              n_output_frames=fp['n_output_frames'],
                                              copyvars=['_path_'])

        if features is None:
            raise DLPyError('cannot extract audio features')

        if cls.running_caslib is None:
            raise DLPyError('there is something wrong, cannot identify the current caslib')

        me = cls.load_audio_metadata(conn, metadata_path,
                                     audio_path=cls.running_caslib)
        if me is None:
            raise DLPyError('cannot load the audio metadata')

        conn.loadactionset('deeplearn', _messagelevel='error')

        if casout is None:
            casout = dict(name=random_name('AudioTable', 6))
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if 'name' not in casout:
            casout['name'] = random_name('AudioTable', 6)

        rt = conn.retrieve('dlJoin', _messagelevel='error', casout=casout, annotation=me, table=features, id='_path_')

        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('cannot create the final audio table!')

        return AudioTable(casout['name'])
