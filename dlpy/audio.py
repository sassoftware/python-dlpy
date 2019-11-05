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

from dlpy import ImageTable
from dlpy.speech_utils import convert_audio_files, convert_audio_files_to_specgrams
from .data_clean import DataClean
from .utils import random_name, find_caslib, get_cas_host_type, find_path_of_caslib
from dlpy.utils import DLPyError


class AudioTable(CASTable):
    ''' CAS Table for Audio '''

    running_caslib = None
    feature_size = None
    num_of_frames_col = None
    label_col = None

    @classmethod
    def load_audio_files(cls, conn, path=None, casout=None, caslib=None,
                         local_audio_path=None, server_audio_path=None,
                         as_specgram=False, label_level=0):
        '''
        Load audio files from path

        Parameters
        ----------
        conn : CAS
            CAS connection object
        path : string, optional
            Path to the audio listing file. This path must user the server-side style and can be access by the server.
        casout : dict, string, or CASTable, optional
            The output CAS table specification
        caslib : string, optional
            The caslib to load audio files from
        local_audio_path : string, optional
            The local location that contains all the audio files.
            This path is on the client side and is host-dependent and must be accessible on the server side.
            When this path is specified, the path option will be ignored.
            All the audio files that are supported by soundfile
            will be first converted into the wave files (i.e., 1 channel, 16 bits, 16K Hz) and then will be loaded into
            the server.
            When caslib is specified, the contents under local_audio_path must be accessible with the given caslib.
            When caslib is None,
            server_audio_path (if specified) will be used to check whether any existing caslib on the server
            can be mapped back to local_audio_path and to create the caslib.
            If no caslib can be found, a new caslib with a random name will
            be generated with local_audio_path.
            This option requires VDMML 8.5 at least.
        server_audio_path : string, optional
            The server location that contains all the audio files. Both local_audio_path and server_audio_path point to
            the same physical location. They could be same if the client and server run on the same machine.
            When caslib is specified, the server_audio_path option will be ignored.
            When caslib is None and the existing caslibs on the server contain the server_audio_path,
            the corresponding caslib will be used. Otherwise, a caslib with a random name will be created.
        as_specgram: bool, optional
            when True, the audio files under local_audio_path will be converted into PNG files
            that contain spectrograms and will be loaded as images into the server.
            Default: False
        label_level : optional
            Specifies which path level should be used to generate the class labels for each image
            when as_specgram is True.
            For instance, label_level = 1 means the first directory and label_level = -2 means the last directory.
            This internally use the SAS scan function
            (check https://www.sascrunch.com/scan-function.html for more details).
            In default, no class labels are generated. If the label column already exists, this option will be ignored.
            Default: 0

        Returns
        -------
        :class:`AudioTable` or `ImageTable`
            If audio files are found, return AudioTable. When as_specgram is True, return ImageTable
        None
            If no audio files are found

        '''

        # check the options

        # either path or local_audio_path needs be specified
        if path is None and local_audio_path is None:
            raise DLPyError('either path or local_audio_path is required to load audio files.')

        # if path is specified, use the listing to create the audio table
        # if local_audio_path is specified, path will be ignored
        if path and local_audio_path:
            print('WARNING: the path option with {} is ignored.'.format(path))

        # if caslib is specified, server_audio_path will be ignored
        if caslib and local_audio_path and server_audio_path:
            print('WARNING: the server_audio_path option with {} is ignored. Use the caslib option with {}.'
                  .format(server_audio_path, caslib))

        # convert audio files when local_audio_path is specified
        if local_audio_path:
            path = None
            if as_specgram:
                convert_audio_files_to_specgrams(local_audio_path, recurse=True)
            else:
                convert_audio_files(local_audio_path, recurse=True)
        else:
            # as_specgram is only supported when local_audio_path is specified
            as_specgram = False

        if as_specgram:
            file_type = 'IMAGE'
        else:
            file_type = 'AUDIO'

        conn.loadactionset('audio', _messagelevel='error')

        if casout is None:
            casout = dict(name=random_name('AudioTable', 6))
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if caslib is None:
            # get the os of the server
            server_type = get_cas_host_type(conn).lower()
            if path:
                if server_type.startswith("lin") or server_type.startswith("osx"):
                    path_split = path.rsplit("/", 1)
                else:
                    path_split = path.rsplit("\\", 1)
            else:
                path_split = [server_audio_path]

            # try accessing the file
            if len(path_split) == 2 or server_audio_path:
                caslib = find_caslib(conn, path_split[0])
                if caslib is not None:

                    if local_audio_path:
                        # call loadTable directly to load binary audio data
                        rt2 = conn.retrieve('table.loadtable', _messagelevel='error', casout=casout,
                                            caslib=caslib, path='',
                                            importOptions=dict(fileType=file_type, contents=True, recurse=True))
                    else:
                        rt2 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                            caslib=caslib, path=path_split[1])
                    if rt2.severity > 1:
                        for msg in rt2.messages:
                            print(msg)
                        raise DLPyError('cannot load audio files, something is wrong!')
                    cls.running_caslib = path_split[0]
                    if as_specgram:
                        out = ImageTable.from_table(conn.CASTable(casout['name']), columns=['_path_'],
                                                    label_level=label_level, casout=casout)
                    else:
                        out = AudioTable(casout['name'])
                    out.set_connection(connection=conn)
                    return out
                else:
                    caslib = random_name('Caslib', 6)
                    rt2 = conn.retrieve('addcaslib', _messagelevel='error', name=caslib, path=path_split[0],
                                        activeonadd=False, subdirectories=True, datasource={'srctype':'path'})
                    if rt2.severity < 2:
                        if local_audio_path:
                            # call loadTable directly to load binary audio data
                            rt3 = conn.retrieve('table.loadtable', _messagelevel='error', casout=casout,
                                                caslib=caslib, path='',
                                                importOptions=dict(fileType=file_type, contents=True, recurse=True))
                        else:
                            rt3 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                                caslib=caslib, path=path_split[1])
                        if rt3.severity > 1:
                            for msg in rt3.messages:
                                print(msg)
                            raise DLPyError('cannot load audio files, something is wrong!')
                        else:
                            cls.running_caslib = path_split[0]
                            if as_specgram:
                                out = ImageTable.from_table(conn.CASTable(casout['name']), columns=['_path_'],
                                                            label_level=label_level, casout=casout)
                            else:
                                out = AudioTable(casout['name'])
                            out.set_connection(connection=conn)
                            return out
            else:
                print("WARNING: Specified path could not be reached. Make sure that the path is accessible by"
                      " the CAS server.")
            return None
        else:

            if local_audio_path:
                # call loadTable directly to load binary audio data
                rt4 = conn.retrieve('table.loadtable', _messagelevel='error', casout=casout,
                                    caslib=caslib, path='',
                                    importOptions=dict(fileType=file_type, contents=True, recurse=True))
            else:
                rt4 = conn.retrieve('audio.loadaudio', _messagelevel='error', casout=casout,
                                    caslib=caslib, path=path)
            if rt4.severity > 1:
                for msg in rt4.messages:
                    print(msg)
                raise DLPyError('cannot load audio files, something is wrong!')
            cls.running_caslib = find_path_of_caslib(conn, caslib)
            if as_specgram:
                out = ImageTable.from_table(conn.CASTable(casout['name']), columns=['_path_'],
                                            label_level=label_level, casout=casout)
            else:
                out = AudioTable(casout['name'])
            out.set_connection(connection=conn)
            return out

    # private function
    @staticmethod
    def __extract_audio_features(conn, table, frame_shift=10, frame_length=25, n_bins=40, n_ceps=40,
                                 feature_scaling_method='STANDARDIZATION', n_output_frames=500, casout=None,
                                 label_level=0,
                                 random_shuffle=True,
                                 **kwargs):

        conn.loadactionset('audio', _messagelevel='error')

        if isinstance(table, AudioTable) is False and isinstance(table, CASTable) is False:
            return None

        if casout is None:
            casout = dict(name=random_name('AudioTable', 6))
        elif isinstance(casout, CASTable) or isinstance(casout, AudioTable):
            casout = casout.to_outtable_params()

        # always use dither with 0 to turn it off
        rt = conn.retrieve('audio.computefeatures', _messagelevel='error', table=table,
                           frameExtractionOptions=dict(frameshift=frame_shift, framelength=frame_length, dither=0.0),
                           melBanksOptions=dict(nbins=n_bins), mfccOptions=dict(nceps=n_ceps),
                           featureScalingMethod=feature_scaling_method, nOutputFrames=n_output_frames,
                           casout=casout, **kwargs)
        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            return None

        server_type = get_cas_host_type(conn).lower()
        if server_type.startswith("lin") or server_type.startswith("osx"):
            fs = "/"
        else:
            fs = "\\"

        if label_level:
            scode = "i=find(_path_,'{0}',-length(_path_)); ".format(fs)
            scode += "length _fName_ varchar(*); length _label_ varchar(*); "
            scode += "_fName_=substr(_path_, i+length('{0}'), length(_path_)-i);".format(fs)
            scode += "_label_=scan(_path_,{},'{}');".format(label_level, fs)
            ctable = CASTable(casout['name'], computedvars=['_fName_', '_label_'],
                              computedvarsprogram=scode)
        else:
            scode = "i=find(_path_,'{0}',-length(_path_)); ".format(fs)
            scode += "length _fName_ varchar(*); "
            scode += "_fName_=substr(_path_, i+length('{0}'), length(_path_)-i);".format(fs)
            ctable = CASTable(casout['name'], computedvars=['_fName_'],
                              computedvarsprogram=scode)

        if random_shuffle:
            conn.table.shuffle(table=ctable, casout=dict(name=casout['name'], replace=True))
        else:
            conn.table.partition(table=ctable, casout=dict(name=casout['name'], replace=True))

        out = AudioTable(casout['name'])
        out.set_connection(connection=conn)

        out.feature_size = n_ceps
        out.num_of_frames_col = '_num_frames_'

        if label_level:
            out.label_col = '_label_'
        else:
            out.label_col = None

        return out

    @classmethod
    def extract_audio_features(cls, conn, table, frame_shift=10, frame_length=25, n_bins=40, n_ceps=40,
                               feature_scaling_method='STANDARDIZATION', n_output_frames=500, casout=None,
                               random_shuffle=True, **kwargs):
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
        random_shuffle : bool, optional
            Specifies whether shuffle the generated CAS table randomly.
            Default: True
        kwargs : keyword-arguments, optional
            Additional parameter for feature extraction.

        Returns
        -------
        :class:`AudioTable`
            If table exists
        None
            If no table exists

        '''

        return cls.__extract_audio_features(conn, table, frame_shift, frame_length, n_bins, n_ceps,
                                            feature_scaling_method, n_output_frames, casout, 0,
                                            random_shuffle, **kwargs)

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

        if conn is None:
            conn = cls.get_connection()

        if conn is None:
            raise DLPyError('cannot get a connection object to the current session.')

        output_name = random_name('AudioTable_Metadata', 6)
        
        dc = DataClean(conn=conn, contents_as_path=path)
        dc_response = dc.process_contents(audio_path = audio_path)
        tbl = dc.create_castable(dc_response['results'], output_name, replace=True, promote=False,
                                 col_names=dc_response['col_names'])

        scode = 'length _fName_ varchar(*); '
        scode += '_fName_ = _filename_; '

        ctbl = CASTable(tbl, computedvars=['_fName_'],
                        computedvarsprogram=scode)

        conn.table.partition(table=ctbl, casout=dict(name=tbl, replace=True))

        return CASTable(tbl)

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

        if conn is None:
            conn = cls.get_connection()

        if conn is None:
            raise DLPyError('cannot get a connection object to the current session.')

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

        rt = conn.retrieve('dlJoin', _messagelevel='error',
                           casout=casout,
                           annotation=me,
                           table=features,
                           id='_fName_')

        if rt.severity > 1:
            for msg in rt.messages:
                print(msg)
            raise DLPyError('cannot create the final audio table!')

        return AudioTable(casout['name'])

    def create_audio_feature_table(self, frame_shift=10, frame_length=25, n_bins=40, n_ceps=40,
                                   feature_scaling_method='STANDARDIZATION', n_output_frames=500,
                                   casout=None, label_level=0, random_shuffle=True):
        '''
        Extracts audio features from the audio table and create a new CASTable that contains the features.

        Parameters
        ----------
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
        label_level : optional
            Specifies which path level should be used to generate the class labels for each audio.
            For instance, label_level = 1 means the first directory and label_level = -2 means the last directory.
            This internally use the SAS scan function
            (check https://www.sascrunch.com/scan-function.html for more details).
            In default, no class labels are generated.
            Default: 0
        random_shuffle : bool, optional
            Specifies whether shuffle the generated CAS table randomly.
            Default: True

        Returns
        -------
        :class:`AudioTable`
            If table exists
        None
            If no table exists

        '''

        table = self

        conn = self.get_connection()

        return self.__extract_audio_features(conn, table, frame_shift, frame_length, n_bins, n_ceps,
                                             feature_scaling_method, n_output_frames, casout, label_level,
                                             copyvars=['_path_'], random_shuffle=random_shuffle)

    @property
    def label_freq(self):
        '''
        Summarize the distribution of different classes (labels) in the audio feature table.
        This requires label_level is specified when creating the feature table.

        Returns
        -------
        :class:`pd.Series`
        '''

        out = self._retrieve('simple.freq', table=self, inputs=['_label_'])['Frequency']
        out = out[['FmtVar', 'Level', 'Frequency']]
        out = out.set_index('FmtVar')
        # out.index.name = 'Label'
        out.index.name = None
        out = out.astype('int64')
        return out

    @property
    def feature_vars(self):
        '''
        Create the feature variable list

        Returns
        -------
        :class:`pd.Series`
        '''

        # build regex to search the columns
        import re
        p = re.compile('_f\d*_v\d*_')

        out = self._retrieve('table.columninfo', table=self)['ColumnInfo']['Column']

        # create the feature variable list
        out_var = []
        for index, value in out.items():
            if p.match(value):
                out_var.append(value)

        return out_var

    @classmethod
    def from_audio_sashdat(cls, conn, path, casout=None):

        '''

        Create an AudioTable from a sashdat file

        Parameters
        ----------
        conn : CAS
            CAS connection object
        path : string, optional
            Path to the audio sashdat file. This path must user the server-side style and can be access by the server.
        casout : dict, string, or CASTable, optional
            The output CAS table specification. When not given, a CASTable with a random name will be generated.

        Returns
        -------
        :class:`AudioTable`

        '''

        if casout is None:
            casout = dict(name=random_name('AudioTable', 6))
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

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
                rt2 = conn.retrieve('table.loadtable', _messagelevel='error', casout=casout,
                                    caslib=caslib, path=path_split[1])
                if rt2.severity > 1:
                    for msg in rt2.messages:
                        print(msg)
                    raise DLPyError('cannot load the audio sashdat file {}, something is wrong!'.format(path))
                cls.running_caslib = path_split[0]
                out = AudioTable(casout['name'])
                out.set_connection(connection=conn)
                return out
            else:
                caslib = random_name('Caslib', 6)
                rt2 = conn.retrieve('addcaslib', _messagelevel='error', name=caslib, path=path_split[0],
                                    activeonadd=False, subdirectories=True, datasource={'srctype':'path'})
                if rt2.severity < 2:
                    # call loadTable directly to load binary audio data
                    rt3 = conn.retrieve('table.loadtable', _messagelevel='error', casout=casout,
                                        caslib=caslib, path=path_split[1])
                    if rt3.severity > 1:
                        for msg in rt3.messages:
                            print(msg)
                        raise DLPyError('cannot load the audio sashdat file {}, something is wrong!'.format(path))
                    else:
                        cls.running_caslib = path_split[0]
                        out = AudioTable(casout['name'])
                        out.set_connection(connection=conn)
                        return out
        else:
            print("WARNING: Specified path could not be reached. Make sure that the path is accessible by"
                  " the CAS server.")
            return None

