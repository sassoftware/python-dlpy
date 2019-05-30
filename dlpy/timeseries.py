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

''' Timeseries related classes and functions '''

from __future__ import (print_function, division, absolute_import, unicode_literals)
from swat.cas.table import CASTable
from .utils import random_name, get_cas_host_type, char_to_double, int_to_double
from dlpy.utils import DLPyError
from swat.cas import datamsghandlers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
import numbers
import re
import swat


def plot_timeseries(tbl, timeid, timeseries, figure=None, 
                    groupid=None, start_time=None, end_time=None, xlim=None, 
                    ylim=None, xlabel=None, ylabel=None, xdate_format=None,
                    title=None, figsize=None, 
                    fontsize_spec=None, **kwargs):
    '''
    Create an timeseries line plot from a CASTable or pandas DataFrame

    Parameters
    ----------
    tbl : :class:`CASTable` or :class:`pandas.DataFrame` or :class:`pandas.Series`
        The input table for the plot. If it is CASTable, it will be fetched to 
        the client. If it is pandas.Series, the index name will become timeid, 
        the series name will become timeseries. 
    timeid : str
        The name of the timeid variable. It will be the value to be used in the 
        x-axis.
    timeseries : str
        The name of the column contains the timeseries value. It will be the
        value to be used in the y-axis.
    figure : two-element-tuple, optional
        The tuple must be in the form (:class:`matplotlib.figure.Figure`,
        :class:`matplotlib.axes.Axes`). These are the figure and axes that the
        user wants to plot on. It can be used to plot new timeseries plot on
        pre-existing figures.
        Default: None
    groupid : dict, optional
        It is in the format {column1 : value1, column2 : value2, ...}.
        It is used to plot subset of the data where column1 = value1 and 
        column2 = value2, etc.
        Default: None, which means do not subset the data.
    start_time : :class:`datetime.datetime` or :class:`datetime.date`, optional
        The start time of the plotted timeseries. 
        Default: None, which means the plot starts at the beginning of the
        timeseries. 
    end_time : :class:`datetime.datetime` or :class:`datetime.date`, optional
        The end time of the plotted timeseries.
        Default: None, which means the plot ends at the end of the timeseries.
    xlim : tuple, optional
        Set the data limits for the x-axis.
        Default: None
    ylim : tuple, optional
        Set the data limits for the y-axis.
        Default: None
    xlabel : string, optional
        Set the label for the x-axis.
    ylabel : string, optional
        Set the label for the y-axis.
    xdate_format : string, optional
        If the x-axis represents date or datetime, this is the date or datetime 
        format string. (e.g. '%Y-%m-%d' is the format of 2000-03-10, 
        refer to documentation for :meth:`datetime.datetime.strftime`)
        Default: None
    title : string, optional
        Set the title of the figure.
        Default: None
    figsize : tuple, optional
        The size of the figure.
        Default: None
    fontsize_spec : dict, optional
        It specifies the fontsize for 'xlabel', 'ylabel', 'xtick', 'ytick', 
        'legend' and 'title'. (e.g. {'xlabel':14, 'ylabel':14}).
        If None, and figure is specified, then it will take from provided
        figure object. Otherwise, it will take the default fontsize, which are
        {'xlabel':16, 'ylabel':16, 'xtick':14, 'ytick':14, 'legend':14, 'title':20}
        Default: None
    `**kwargs` : keyword arguments, optional
        Options to pass to matplotlib plotting method.    

    Returns
    -------
    (:class:`matplotlib.figure.Figure`, :class:`matplotlib.axes.Axes`)

    '''
    default_fontsize_spec = {'xlabel':16, 'ylabel':16, 'xtick':14,
                             'ytick':14, 'legend':14, 'title':20}
    
    if figure is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        if fontsize_spec is not None:
            default_fontsize_spec.update(fontsize_spec)
            
        fontsize_spec = default_fontsize_spec
    else:
        fig, ax = figure
        if fontsize_spec is None:
            fontsize_spec = {}
            
        if 'legend' not in fontsize_spec.keys():
            fontsize_spec['legend'] = default_fontsize_spec['legend']
        
    if isinstance(tbl, CASTable):
        if groupid is None:
            tbl = tbl.to_frame()
        else:
            where_clause_list = []
            for gid in groupid.keys():
                where_clause_list.append(gid + '=' + str(groupid[gid]))
                
            where_clause = ' and '.join(where_clause_list)
            tbl = tbl.query(where_clause)
            tbl = tbl.to_frame()
    else:
        if isinstance(tbl, pd.Series):
            timeseries = tbl.name
            tbl = tbl.reset_index()
            timeid = [colname for colname in tbl.columns if colname != timeseries][0]
            
        if groupid is not None:
            for gid in groupid.keys():
                tbl = tbl.loc[tbl[gid]==groupid[gid]]

    if not (np.issubdtype(tbl[timeid].dtype, np.integer) or
            np.issubdtype(tbl[timeid].dtype, np.floating)):
        tbl[timeid] = pd.to_datetime(tbl[timeid])
        fig.autofmt_xdate()
        if xdate_format is not None:
            import matplotlib.dates as mdates
            xfmt = mdates.DateFormatter(xdate_format)
            ax.xaxis.set_major_formatter(xfmt)
                
    if start_time is not None:
        if isinstance(start_time, datetime.date):
            start_time = pd.Timestamp(start_time)
            
        tbl = tbl.loc[tbl[timeid]>=start_time]
    
    if end_time is not None:
        if isinstance(start_time, datetime.date):
            end_time = pd.Timestamp(end_time)
            
        tbl = tbl.loc[tbl[timeid]<=end_time]
        
    tbl = tbl.sort_values(timeid)
               
    ax.plot(tbl[timeid], tbl[timeseries], **kwargs)
    
    if xlabel is not None:    
        if 'xlabel' in fontsize_spec.keys():
            ax.set_xlabel(xlabel, fontsize=fontsize_spec['xlabel'])
        else:
            ax.set_xlabel(xlabel)
    elif figure is not None:
        if 'xlabel' in fontsize_spec.keys():
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize_spec['xlabel'])
    else:
        ax.set_xlabel(timeid, fontsize=fontsize_spec['xlabel'])
            
        
    if ylabel is not None:
        if 'ylabel' in fontsize_spec.keys():
            ax.set_ylabel(ylabel, fontsize=fontsize_spec['ylabel'])
        else:
            ax.set_ylabel(ylabel)
    elif figure is not None:
        if 'ylabel' in fontsize_spec.keys():
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize_spec['ylabel'])
    else:
        ax.set_ylabel(timeseries, fontsize=fontsize_spec['ylabel'])
    
    if xlim is not None:    
        ax.set_xlim(xlim)
        
    if ylim is not None:
        ax.set_ylim(ylim)
        
    if title is not None:
        if 'title' in fontsize_spec.keys():
            ax.set_title(title, fontsize=fontsize_spec['title'])
        else:
            ax.set_title(title)
    elif figure is not None:
        if 'title' in fontsize_spec.keys():
            ax.set_title(ax.get_title(), fontsize=fontsize_spec['title'])
        
    ax.legend(loc='best', bbox_to_anchor=(1, 1), prop={'size': fontsize_spec['legend']})
    if 'xtick' in fontsize_spec.keys():
        ax.get_xaxis().set_tick_params(direction='out', labelsize=fontsize_spec['xtick'])
    else:
        ax.get_xaxis().set_tick_params(direction='out')
        
    if 'ytick' in fontsize_spec.keys():
        ax.get_yaxis().set_tick_params(direction='out', labelsize=fontsize_spec['ytick'])
    else:
        ax.get_yaxis().set_tick_params(direction='out') 
    
     
    return (fig, ax)
        

class TimeseriesTable(CASTable):
    '''
    Table for preprocessing timeseries

    It creates an instance of :class:`TimeseriesTable` by loading from
    files on the server side, or files on the client side, or in
    memory :class:`CASTable`, :class:`pandas.DataFrame` or
    :class:`pandas.Series. It then performs inplace timeseries formatting,
    timeseries accumulation, timeseries subsequence generation, and
    timeseries partitioning to prepare the timeseries into a format that
    can be followed by subsequent deep learning models.
    
    Parameters
    ----------
    name : string, optional
        Name of the CAS table
    timeid : string, optional
        Specifies the column name for the timeid. 
        Default: None
    groupby_var : string or list-of-strings, optional
        The groupby variables. 
        Default: None.
    sequence_opt : dict, optional
        Dictionary with keys: 'input_length', 'target_length' and 'token_size'.
        It will be created by the prepare_subsequences method.
        Default: None
    inputs_target : dict, optional
        Dictionary with keys: 'inputs', 'target'.
        It will be created by the prepare_subsequences method.
        Default: None

    Attributes
    ----------
    timeid_type : string
         Specifies whether the table uses 'date' or 'datetime' format

    Returns
    -------
    :class:`TimeseriesTable`

    '''
    running_caslib = None

    def __init__(self, name, timeid=None, groupby_var=None,  
                 sequence_opt=None, inputs_target=None, target=None,
                 autoregressive_sequence=None, acc_interval=None,
                 **table_params):
        CASTable.__init__(self, name, **table_params)
        self.timeid = timeid
        self.groupby_var = groupby_var                   
        self.sequence_opt = sequence_opt
        self.inputs_target = inputs_target
        self.target = target
        self.autoregressive_sequence=autoregressive_sequence
        self.acc_interval=acc_interval


    @classmethod
    def from_table(cls, tbl, columns=None, casout=None):
        '''
        Create an TimeseriesTable from a CASTable

        Parameters
        ----------
        tbl : :class:`CASTable`
            The CASTable object to use as the source.
        columns : list-of-strings, optional
            Columns to keep when loading the data.
            None means it will include all the columns from the source.
            Empty list means include no column, which will generate empty data.
            Default: None
        casout : dict or :class:`CASTable`, optional
            if it is dict, it specifies the output CASTable parameters.
            if it is CASTable, it is the CASTable that will be overwritten. 
            None means a new CASTable with random name will be generated.
            Default: None

        Returns
        -------
        :class:`TimeseriesTable`

        '''
        input_tbl_params = tbl.to_outtable_params()
        input_tbl_name = input_tbl_params['name']

        conn = tbl.get_connection()

        if casout is None:
            casout_params = {}
        elif isinstance(casout, CASTable):
            casout_params = casout.to_outtable_params()
        elif isinstance(casout, dict):
            casout_params = casout

        if 'name' not in casout_params:
            casout_params['name'] = random_name('ts', 4)
            
        output_tbl_name = casout_params['name']
        
        if columns is None:
            keep_col_sascode = '''
            data {0};
            set {1};
            run;
            '''.format(output_tbl_name, input_tbl_name)
        
            conn.retrieve('dataStep.runCode', _messagelevel='error',
                          code=keep_col_sascode)
        else:
            if not isinstance(columns, list):
                columns = [columns]
                
            keepcol = ' '.join(columns)
        
            keep_col_sascode = '''
            data {0};
            set {1};
            keep {2};
            run;
            '''.format(output_tbl_name,  input_tbl_name, keepcol)
        
            conn.retrieve('dataStep.runCode', _messagelevel='error',
                          code=keep_col_sascode)
            
        
        out = cls(**casout_params)
        out.set_connection(conn)

        return out

    @classmethod
    def from_pandas(cls, conn, pandas_df, casout=None):
        '''
        Create an TimeseriesTable from a pandas DataFrame or Series

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        pandas_df : :class:`pandas.DataFrame` or :class:`pandas.Series`
            The pandas dataframe or series to use as the source.
        casout : dict or :class:`CASTable`, optional
            if it is dict, it specifies the output CASTable parameters.
            if it is CASTable, it is the CASTable that will be overwritten. 
            None means a new CASTable with random name will be generated.
            Default: None

        Returns
        -------
        :class:`TimeseriesTable`

        '''
        if isinstance(pandas_df, pd.Series):
            pandas_df = pandas_df.reset_index()

        if casout is None:
            casout_params = {}
        elif isinstance(casout, CASTable):
            casout_params = casout.to_outtable_params()
        elif isinstance(casout, dict):
            casout_params = casout

        if 'name' not in casout_params:
            casout_params['name'] = random_name('ts', 4)
            
        output_tbl_name = casout_params['name']
        
        handler = datamsghandlers.PandasDataFrame(pandas_df)
        
        conn.addtable(table=output_tbl_name, replace=True, **handler.args.addtable)
        
        tbl = conn.CASTable(name=output_tbl_name)
        
        return cls.from_table(tbl, columns=None, casout=casout_params)

    @classmethod
    def from_localfile(cls, conn, path, columns=None, importoptions=None,
                       casout=None):
        '''
        Create an TimeseriesTable from a file on the client side.

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The full path to the local file that will be uploaded to the server.
        columns : list-of-strings, optional
            Columns to keep when loading the data.
            None means it will include all the columns from the source.
            Empty list means to include no column, which will generate empty data.
            Default: None
        importoptions : dict, optional
            Options to import data and upload to the server, such as filetype,
            delimiter, etc. None means use the default 'auto' method in the
            importoptions from CAS.upload.
            Default: None
        casout : dict or :class:`CASTable`, optional
            If it is dict, it specifies the output CASTable parameters.
            If it is CASTable, it is the CASTable that will be overwritten.
            None means a new CASTable with random name will be generated.
            Default: None

        Returns
        -------
        :class:`TimeseriesTable`

        '''
        if casout is None:
            casout_params = {}
        elif isinstance(casout, CASTable):
            casout_params = casout.to_outtable_params()
        elif isinstance(casout, dict):
            casout_params = casout

        if 'name' not in casout_params:
            casout_params['name'] = random_name('ts', 4)
            
        if importoptions is None:
            importoptions = {}
        
        upload_result = conn.upload(path, 
                                    importoptions=importoptions, 
                                    casout=casout_params)
        
        tbl = conn.CASTable(**casout_params)
        
        return cls.from_table(tbl, columns=columns, casout=casout_params)
        
    @classmethod
    def from_serverfile(cls, conn, path, columns=None, caslib=None,
                        importoptions=None, casout=None):
        '''
        Create an TimeseriesTable from a file on the server side

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The path that the server can access. If the caslib is specified,
            it is relative path to the file with respect to the caslib.
            otherwise, it is the full path to the file.
        columns : list-of-strings, optional
            columns to keep when loading the data.
            None means it will include all the columns from the source.
            Empty list means include no column, which will generate empty data.
            Default: None
        caslib : string, optional
            The name of the caslib which contains the file to be uploaded.
            Default: None
        importoptions : dict, optional
            Options to import data and upload to the server, such as filetype,
            delimiter, etc. None means use the default 'auto' method in the
            importoptions from CAS.upload.
            Default: None
        casout : dict or :class:`CASTable`, optional
            If it is dict, it specifies the output CASTable parameters.
            If it is CASTable, it is the CASTable that will be overwritten.
            None means a new CASTable with random name will be generated.
            Default: None

        Returns
        -------
        :class:`TimeseriesTable`

        '''
        if casout is None:
            casout_params = {}
        elif isinstance(casout, CASTable):
            casout_params = casout.to_outtable_params()
        elif isinstance(casout, dict):
            casout_params = casout

        if 'name' not in casout_params:
            casout_params['name'] = random_name('ts', 4)
            
        if importoptions is None:
            importoptions = {}
        
        if caslib is None:
            caslib, rest_path = cls.find_file_caslib(conn, path)
            if caslib is None:
                server_type = get_cas_host_type(conn).lower()
                if server_type.startswith("lin") or server_type.startswith("osx"):
                    path_split = path.rsplit("/", 1)
                else:
                    path_split = path.rsplit("\\", 1)
                    
                caslib = random_name('Caslib', 6)
                rt1 = conn.retrieve('addcaslib', _messagelevel='error', 
                                    name=caslib, path=path_split[0],
                                    activeonadd=False, subdirectories=False, 
                                    datasource={'srctype':'path'})
                
                if rt1.severity < 2:
                    rt2 = conn.retrieve('table.loadTable', 
                                        _messagelevel='error', 
                                        casout=casout_params,
                                        caslib=caslib,
                                        importoptions=importoptions,
                                        path=path_split[1])
                    if rt2.severity > 1:
                        for msg in rt2.messages:
                            print(msg)
                        raise DLPyError('cannot load files, something is wrong!')
                else:
                    for msg in rt1.messages:
                        print(msg)
                    raise DLPyError('''cannot create caslib with path:{}, 
                                    something is wrong!'''.format(path_split[0]))
            else:
                rt3 = conn.retrieve('table.loadTable', 
                    _messagelevel='error', 
                    casout=casout_params,
                    caslib=caslib, 
                    importoptions=importoptions,
                    path=rest_path)
                if rt3.severity > 1:
                    for msg in rt3.messages:
                        print(msg)
                    raise DLPyError('cannot load files, something is wrong!')
        else:
            rt4 = conn.retrieve('table.loadTable', 
                _messagelevel='error', 
                casout=casout_params,
                caslib=caslib, 
                importoptions=importoptions,
                path=path)
            if rt4.severity > 1:
                for msg in rt4.messages:
                    print(msg)
                raise DLPyError('cannot load files, something is wrong!')
                
        
        tbl = conn.CASTable(**casout_params)
        
        return cls.from_table(tbl, columns=columns, casout=casout_params)
    
    
    def timeseries_formatting(self, timeid, timeseries, 
                              timeid_informat=None, timeid_format=None,
                              extra_columns=None):
        '''
        Format the TimeseriesTable

        Format timeid into appropriate format and check and format
        timeseries columns into numeric columns.

        Parameters
        ----------
        timeid : string
            Specifies the column name for the timeid. 
        timeseries : string or list-of-strings
            Specifies the column name for the timeseries, that will be part of 
            the input or output of the RNN. If str, then it is univariate 
            time series. If list of strings, then it is multivariate timeseries.                        
        timeid_informat : string, optional
            if timeid is in the string format, this is required to parse the 
            timeid column. 
            Default: None
        timeid_format : string, optional
            Specifies the SAS format that the timeid column will be stored in
            after parsing.
            None means it will be stored in numeric form, not a specific date or datetime format.
            Default: None
        extra_columns : string or list-of-strings, optional
            Specifies the addtional columns to be included. 
            Empty list means to include no extra columns other than timeid and timeseries.
            if None, all columns are included.
            Default: None

        '''
        self.timeid = timeid
        self.timeseries = timeseries
        self.timeid_format = timeid_format
        self.timeid_informat = timeid_informat   
        self.extra_columns = extra_columns
        
        input_tbl_params = self.to_outtable_params()
        input_tbl_name = input_tbl_params['name']

        conn = self.get_connection()
        
        tbl_colinfo = self.columninfo().ColumnInfo
        
        if self.timeid_format is None:
            if self.timeid_informat is None:
                self.timeid_format = self.timeid_informat
            elif self.timeid_informat.lower().startswith('anydtdtm'):
                self.timeid_format = 'DATETIME19.'
            else:
                self.timeid_format = self.timeid_informat
            

        if (((self.timeid_type not in ['double', 'date', 'datetime']) 
        and (not self.timeid_type.startswith('int'))) 
        and (self.timeid_informat is not None)):
            fmt_code = '''
            data {0}; 
            set {0}(rename=({1}=c_{1})); 
            {1} = input(c_{1},{2});
            drop c_{1};
            format {1} {3};
            run;
            '''.format(input_tbl_name, self.timeid, 
            self.timeid_informat, self.timeid_format)  
            
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=fmt_code)
            
        elif (((self.timeid_type not in ['double', 'date', 'datetime']) 
        and (not self.timeid_type.startswith('int'))) 
        and (self.timeid_informat is None)):
            raise ValueError('''timeid variable is not in the numeric format, 
            so timeid_informat is required for parsing the timeid variable. 
                             ''')
        elif (self.timeid_format is not None):
            fmt_code = '''
            data {0}; 
            set {0}; 
            format {1} {2};
            run;
            '''.format(input_tbl_name, self.timeid, self.timeid_format)            
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=fmt_code)            
        else:
            fmt_code = '''
            data {0}; 
            set {0}; 
            run;
            '''.format(input_tbl_name)            
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=fmt_code)
        
        tbl_colinfo = self.columninfo().ColumnInfo
        
        if not isinstance(self.timeseries, list):
            self.timeseries = [self.timeseries]
        
        if set(self.timeseries).issubset(tbl_colinfo.Column):
            char_to_double(conn, tbl_colinfo, input_tbl_name, 
                               input_tbl_name, self.timeseries)
        else:
            raise ValueError('''One or more variables specified in 'timeseries' 
            do not exist in the input table.
                             ''')
            
        if self.extra_columns is not None:
            if not isinstance(self.extra_columns, list):
                self.extra_columns = [self.extra_columns] 
                
            keepcol = [self.timeid]
            keepcol.extend(self.timeseries + self.extra_columns)
            keepcol = ' '.join(keepcol)
            
            keep_col_sascode = '''
            data {0};
            set {0};
            keep {1};
            run;
            '''.format(input_tbl_name, keepcol)
            
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=keep_col_sascode)
            
        print('NOTE: Timeseries formatting is completed.')
        
    def timeseries_accumlation(self, acc_interval='day',timeid=None,
                               timeseries=None, groupby=None,
                               extra_num_columns=None, default_ts_acc='sum',
                               default_col_acc = 'avg',
                               acc_method_byvar=None):
        '''
        Accumulate the TimeseriesTable into regular consecutive intervals

        Parameters
        ----------
        acc_interval : string, optional
            The accumulation interval, such as 'year', 'qtr', 'month', 'week',
            'day', 'hour', 'minute', 'second'.            
        timeid : string, optional
            Specifies the column name for the timeid. 
            If None, it will take the timeid specified in timeseries_formatting.
            Default: None
        timeseries : string or list-of-strings, optional
            Specifies the column name for the timeseries, that will be part of 
            the input or output of the RNN. If str, then it is univariate 
            time series. If list of strings, then it is multivariate timeseries. 
            If None, it will take the timeseries specified in timeseries_formatting.
            Default: None
        groupby : string or list-of-strings, optional
            The groupby variables. 
            Default: None
        extra_num_columns : string or list-of-strings, optional
            Specifies the addtional numeric columns to be included for
            accumulation. These columns can include static feature, and might
            be accumulated differently than the timeseries that will be used
            in RNN. if None, it means no additional numeric columns will be
            accumulated for later processing and modeling.
            Default: None
        default_ts_acc : string, optional
            Default accumulation method for timeseries.
            Default: sum
        default_col_acc : string, optional
            Default accumulation method for additional numeric columns
            Default: avg
        acc_method_byvar : dict, optional
            It specifies specific accumulation method for individual columns,
            if the method is different from the default.
            It has following structure: {'column1 name': 'accumulation method1',
            'column2 name': 'accumulation method2', ...}
            Default: None

        '''
        if (timeid is None) and (self.timeid is None):
            raise DLPyError('''timeid is not specified, consider specifying 
            and formatting it with timeseries_formatting''')
        elif (timeid is not None) and (timeid != self.timeid):
            warnings.warn('''timeid has not been formatted by timeseries_formatting,
            consider reload the data and use timeseries_formatting to format the data,
            unless the data has already been pre-formatted.''')
            self.timeid = timeid

        if timeseries is None:
            if ((hasattr(self, 'timeseries') and self.timeseries is None) or 
                (not hasattr(self, 'timeseries'))):                
                raise DLPyError('''timeseries is not specified, consider specifying 
                and formatting it with timeseries_formatting''')
        else:
            if not isinstance(timeseries, list):
                timeseries = [timeseries]
                
            if ((hasattr(self, 'timeseries') and (self.timeseries is None)) or 
                (not hasattr(self, 'timeseries'))): 
                warnings.warn('''timeseries has not been formatted by timeseries_formatting,
                consider reload the data and use timeseries_formatting to format the data,
                unless the data has already been pre-formatted.''')
            elif not set(timeseries).issubset(self.timeseries):
                warnings.warn('''timeseries contains variable(s) that has not been
                formatted by timeseries_formatting, consider reload the data and use 
                timeseries_formatting to format the data,
                unless the data has already been pre-formatted.''')

            self.timeseries = timeseries

        self.groupby_var = groupby
        self.extra_num_columns = extra_num_columns
        
        input_tbl_params = self.to_outtable_params()
        input_tbl_name = input_tbl_params['name']

        conn = self.get_connection()
        conn.loadactionset('timeData')
        
        tbl_colinfo = self.columninfo().ColumnInfo
        
        if self.groupby_var is None:
            self.groupby_var = []
        elif not isinstance(self.groupby_var, list):
            self.groupby_var = [self.groupby_var]
        
        if set(self.groupby_var).issubset(tbl_colinfo.Column):
            int_to_double(conn, tbl_colinfo, input_tbl_name, 
                               input_tbl_name, self.groupby_var)
        else:
            raise ValueError('''One or more variables specified in 'groupby' 
            do not exist in the input table.
                             ''')
            
        tbl_colinfo = self.columninfo().ColumnInfo 
        
        #Check timeid is in the input columns
        if self.timeid not in tbl_colinfo.Column.values:
            raise ValueError('''variable 'timeid' does not exist in input table.
                             ''')
        
        #Check timeseries is in the input columns
        if not isinstance(self.timeseries, list):
            self.timeseries = [self.timeseries]
        
        if not set(self.timeseries).issubset(tbl_colinfo.Column):
            raise ValueError('''One or more variables specified in 'timeseries' 
            do not exist in the input table.
                             ''')
            
        #Check extra_num_columns is in the input columns    
        if self.extra_num_columns is None:
            self.extra_num_columns = []
        elif not isinstance(self.extra_num_columns, list):
            self.extra_num_columns = [self.extra_num_columns]
        
        if not set(self.extra_num_columns).issubset(tbl_colinfo.Column):
            raise ValueError('''One or more variables specified in 'extra_num_columns' 
            do not exist in the input table.
                             ''')

        if self.timeid_type == 'datetime':
            acc_interval = 'dt' + acc_interval
        elif ((self.timeid_type == 'date') 
        and (acc_interval.lower() in ['hour', 'minute', 'second'])):
            raise ValueError('''the acc_interval has higher frequency than day, 
            yet the timeid variable is in the date format. 
                             ''')   
            
        self.acc_interval = acc_interval

        if acc_method_byvar is None:
            acc_method_byvar = {}
        
        serieslist = []
        for ts in self.timeseries:
            if ts in acc_method_byvar.keys():
                method_dict = {'acc':acc_method_byvar[ts],'name':ts}
                serieslist.append(method_dict)
            else:
                method_dict = {'acc':default_ts_acc,'name':ts}
                serieslist.append(method_dict)
        
        for extra_col in self.extra_num_columns:
            if extra_col in self.timeseries:
                warnings.warn('''
                              columns in extra_num_columns are also found in 
                              timeseries, and will be ignored.
                              ''')
                continue
            
            elif extra_col in acc_method_byvar.keys():
                method_dict = {'acc':acc_method_byvar[extra_col],'name':extra_col}
                serieslist.append(method_dict)
            else:
                method_dict = {'acc':default_col_acc,'name':extra_col}
                serieslist.append(method_dict)
                
        acc_result = conn.retrieve('timedata.timeseries', _messagelevel='error',
            table={'groupby':self.groupby_var,'name': input_tbl_name},
            series=serieslist,
            timeid=self.timeid,
            interval=self.acc_interval,
            trimid='BOTH',
            sumout=dict(name=input_tbl_name + '_summary', replace=True),
            casout=dict(name=input_tbl_name, replace=True))
        
        if self.acc_interval.startswith('dt'):            
            print('NOTE: Timeseries are accumulated to the frequency of {}'.format(self.acc_interval[2:]))
        else:
            print('NOTE: Timeseries are accumulated to the frequency of {}'.format(self.acc_interval))
        
    def prepare_subsequences(self, seq_len, target, predictor_timeseries=None,
                             timeid=None, groupby=None,
                             input_length_name='xlen', target_length_name='ylen',
                             missing_handling='drop'):
        '''
        Prepare the subsequences that will be pass into RNN

        Parameters
        ----------
        seq_len : int
            subsequence length that will be passed onto RNN.
        target : string
            the target variable for RNN. Currenly only support univariate target,
            so only string is accepted here, not list of strings.
        predictor_timeseries : string or list-of-strings, optional
            Timeseries that will be used to predict target. They will be preprocessed
            into subsequences as well. If None, it will take the target timeseries
            as the predictor, which corresponds to auto-regressive models.
            Default: None
        timeid : string, optional
            Specifies the column name for the timeid. 
            If None, it will take the timeid specified in timeseries_accumlation.
            Default: None
        groupby : string or list-of-strings, optional
            The groupby variables. if None, it will take the groupby specified
            in timeseries_accumlation.
            Default: None
        input_length_name : string, optional
            The column name in the CASTable specifying input sequence length. 
            Default: xlen
        target_length_name : string, optional
            The column name in the CASTable specifying target sequence length. 
            currently target length only support length 1 for numeric sequence.
            Default: ylen
        missing_handling : string, optional
            How to handle missing value in the subsequences. 
            default: drop

        '''        
        tbl_colinfo = self.columninfo().ColumnInfo
        input_tbl_params = self.to_outtable_params()
        input_tbl_name = input_tbl_params['name']

        conn = self.get_connection()
        
        if timeid is not None:
            self.timeid = timeid
        elif self.timeid is None:
            raise ValueError('''timeid is not specified''')
        
        if self.timeid not in tbl_colinfo.Column.values:
            raise ValueError('''timeid does not exist in the input table''')
            
        if groupby is not None:
            self.groupby_var = groupby
            
        if self.groupby_var is None:
            self.groupby_var = []
        elif not isinstance(self.groupby_var, list):
            self.groupby_var = [self.groupby_var]
        
        if set(self.groupby_var).issubset(tbl_colinfo.Column):
            int_to_double(conn, tbl_colinfo, input_tbl_name, 
                               input_tbl_name, self.groupby_var)
        else:
            raise ValueError('''One or more variables specified in 'groupby' 
            do not exist in the input table.
                             ''')
        
        if isinstance(target, list):
            if len(target) > 1:
                raise DLPyError('''currently only support univariate target''')
        else:
            target = [target]
                
        if predictor_timeseries is None:
            predictor_timeseries = target
        elif not isinstance(predictor_timeseries, list):
            predictor_timeseries = [predictor_timeseries]
            
        if set(target).issubset(predictor_timeseries):
            independent_pred = [var for var in predictor_timeseries 
                                if var not in target]            
            self.auto_regressive = True
        else:
            independent_pred = predictor_timeseries
            self.auto_regressive = False
            
        if not set(target).issubset(tbl_colinfo.Column):
            raise ValueError('''invalid target variable''')
        
        if len(independent_pred) > 0:
            if not set(independent_pred).issubset(tbl_colinfo.Column):
                raise ValueError('''columns in predictor_timeseries are absent from
                                 the accumulated timeseriest table.''') 
        
        if self.timeseries is None:
            warnings.warn('''timeseries has not been formatted by timeseries_formatting,
            consider reload the data and use timeseries_formatting to format the data,
            unless the data has already been pre-formatted.''')
        else:
            if not set(target).issubset(self.timeseries):
                warnings.warn('''target is not in pre-formatted timeseries,
                consider reload the data and use timeseries_formatting to format the data,
                unless the data has already been pre-formatted.''')
            
            if len(independent_pred) > 0:
                if not set(independent_pred).issubset(self.timeseries):
                    warnings.warn('''
                                  some of predictor_timeseries are not in pre-accumulated timeseries,\n
                                  consider reload the data and use timeseries_accumulation to accumulate the data,\n
                                  unless the data has already been pre-formatted.
                                  ''')
            
        self.target = target[0]
        self.independent_pred = independent_pred
        self.seq_len = seq_len
        
        if self.seq_len < 1:
            raise ValueError('''RNN sequence length at least need to be 1''')     
        
        sascode = 'data {0}; set {0}; by {1} {2};'.format(
                input_tbl_name, ' '.join(self.groupby_var), self.timeid)
        
        if self.seq_len > 1:
            for var in self.independent_pred:
                sascode += self.create_lags(var, self.seq_len - 1, self.groupby_var)
        
        if self.auto_regressive:
            sascode += self.create_lags(self.target, self.seq_len, self.groupby_var)
         
        sascode += '{0} = {1};'.format(input_length_name, self.seq_len)
        sascode += '{} = 1;'.format(target_length_name) # Currently only support one timestep numeric output.
        if missing_handling == 'drop':
            sascode += 'if not cmiss(of _all_) then output {};'.format(input_tbl_name)
        sascode += 'run;'
        if len(self.groupby_var) == 0:
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode, 
                           single='Yes')
        else:
            conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
            
        self.input_vars = []
        self.autoregressive_sequence = []
        
        for i in range(self.seq_len):
            if self.auto_regressive:
                self.input_vars.append('{0}_lag{1}'.format(self.target, i+1))
                self.autoregressive_sequence.append('{0}_lag{1}'.format(self.target, i+1))
            
            for var in self.independent_pred:
                if i == 0:
                    self.input_vars.append(var)
                else:
                    self.input_vars.append('{0}_lag{1}'.format(var, i))
        
        self.input_vars.reverse()
        self.autoregressive_sequence.reverse()
                    
        self.tokensize = len(predictor_timeseries)
        
        self.sequence_opt = dict(input_length=input_length_name, 
                     target_length=target_length_name,
                     token_size=self.tokensize)
        
        self.inputs_target = dict(inputs=self.input_vars, 
                                  target=self.target)
        
        print('NOTE: timeseries subsequences are prepared with subsequence length = {}'.format(seq_len))
        
    @property
    def timeid_type(self):
        tbl_colinfo = self.columninfo().ColumnInfo
        timeid_type = self.identify_coltype(self.timeid, tbl_colinfo)
        return timeid_type
    
    @staticmethod
    def identify_coltype(col, tbl_colinfo):
        if col not in tbl_colinfo.Column.values:
            raise ValueError('''variable {} does not exist in input table.
                             '''.format(col))
        
        if 'Format' in tbl_colinfo.columns:
            cas_timeid_fmt = tbl_colinfo.Format[tbl_colinfo.Column == col].values[0]
        else:
            cas_timeid_fmt = None
            
        col_type = tbl_colinfo.Type[tbl_colinfo.Column == col].values[0]
        if cas_timeid_fmt:
            for pattern in swat.options.cas.dataset.date_formats:
                if re.match(r'{}\Z'.format(pattern), cas_timeid_fmt):
                    col_type = 'date'
                    break
            
            for pattern in swat.options.cas.dataset.datetime_formats:
                if re.match(r'{}\Z'.format(pattern), cas_timeid_fmt):
                    if col_type == 'date':
                        raise DLPyError('''{} format in CASTable is ambiguous,
                        and can match both sas date and sas datetime format'''.format(col))
                    else:                        
                        col_type = 'datetime'
                        break
                    
        return col_type        

    def timeseries_partition(self, training_start=None, validation_start=None, 
                             testing_start=None, end_time=None, 
                             partition_var_name='split_id', 
                             traintbl_suffix='train',
                             validtbl_suffix='valid',
                             testtbl_suffix='test'):
        '''
        Split the dataset into training, validation and testing set

        Parameters
        ----------
        training_start : float or :class:`datetime.datetime` or :class:`datetime.date`, optional
            The training set starting time stamp. if None, the training set
            start at the earliest observation record in the table.
            Default: None
        validation_start : float or :class:`datetime.datetime` or :class:`datetime.date`, optional
            The validation set starting time stamp. The training set
            ends right before it. If None, there is no validation set,
            and the training set ends right before the start of
            testing set.
            Default: None
        testing_start : float or :class:`datetime.datetime` or :class:`datetime.date`, optional
            The testing set starting time stamp. The validation set 
            (or training set if validation set is not specified) ends
            right before it. If None, there is no testing set, and
            the validation set (or training set if validation set is
            not set) ends at the end_time.
            Default: None
        end_time : float or :class:`datetime.datetime` or :class:`datetime.date`, optional
            The end time for the table.
        partition_var_name : string, optional
            The name of the indicator column that indicates training,
            testing and validation.
            Default: 'split_id'.
        traintbl_suffix : string, optional
            The suffix name of the CASTable for the training set.
            Default: 'train'
        validtbl_suffix : string, optional
            The suffix name of the CASTable for the validation set.
            Default: 'valid'
        testtbl_suffix : string, optional
            The suffix name of the CASTable for the testing set.
            Default: 'test'

        Returns
        -------
        ( training TimeseriesTable, validation TimeseriesTable, testing TimeseriesTable )

        '''    
        self.partition_var_name = partition_var_name
        conn = self.get_connection()
                
        training_start = self.convert_to_sas_time_format(training_start, self.timeid_type)
        validation_start = self.convert_to_sas_time_format(validation_start, self.timeid_type)
        testing_start = self.convert_to_sas_time_format(testing_start, self.timeid_type)
        end_time = self.convert_to_sas_time_format(end_time, self.timeid_type)

        if testing_start is None:
            testing_start = end_time
            test_statement = ';'
        else:
            test_statement = self.generate_splitting_code(
                    self.timeid, testing_start, end_time, 
                    True, self.partition_var_name, 'test')

        if validation_start is None:
            validation_start = testing_start
            valid_statement = ';'
        else:
            if testing_start == end_time:
                valid_statement = self.generate_splitting_code(
                        self.timeid, validation_start, testing_start, 
                        True, self.partition_var_name, 'valid') 
            else:
                valid_statement = self.generate_splitting_code(
                        self.timeid, validation_start, testing_start, 
                        False, self.partition_var_name, 'valid')

        if validation_start == end_time:
            train_statement =  self.generate_splitting_code(
                            self.timeid, training_start, validation_start, 
                            True, self.partition_var_name, 'train')
        else:
            train_statement =  self.generate_splitting_code(
                            self.timeid, training_start, validation_start, 
                            False, self.partition_var_name, 'train')
            
        input_tbl_params = self.to_outtable_params()
        input_tbl_name = input_tbl_params['name']
        
        traintbl_name = '_'.join([input_tbl_name, traintbl_suffix])
        validtbl_name = '_'.join([input_tbl_name, validtbl_suffix])
        testtbl_name = '_'.join([input_tbl_name, testtbl_suffix])
        
        splitting_code = '''
        data {4} {5} {6};
        set {0};
        {1}
        {2}
        {3}
        if {7} = 'train' then output {4};
        if {7} = 'valid' then output {5};
        if {7} = 'test' then output {6};
        run;
        '''.format(input_tbl_name, train_statement, valid_statement, test_statement,
        traintbl_name, validtbl_name, testtbl_name, self.partition_var_name)
        
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=splitting_code)
        
        train_out = dict(name=traintbl_name, timeid=self.timeid, groupby_var=self.groupby_var,
                         sequence_opt=self.sequence_opt, inputs_target=self.inputs_target, 
                         target=self.target, autoregressive_sequence = self.autoregressive_sequence, 
                         acc_interval=self.acc_interval)
        
        valid_out = dict(name=validtbl_name, timeid=self.timeid, groupby_var=self.groupby_var,
                         sequence_opt=self.sequence_opt, inputs_target=self.inputs_target,
                         target=self.target, autoregressive_sequence = self.autoregressive_sequence, 
                         acc_interval=self.acc_interval)
                
        test_out = dict(name=testtbl_name, timeid=self.timeid, groupby_var=self.groupby_var,
                         sequence_opt=self.sequence_opt, inputs_target=self.inputs_target, 
                         target=self.target, autoregressive_sequence = self.autoregressive_sequence, 
                         acc_interval=self.acc_interval)  
        
        train_out_tbl = TimeseriesTable(**train_out)
        train_out_tbl.set_connection(conn)
        
        valid_out_tbl = TimeseriesTable(**valid_out)
        valid_out_tbl.set_connection(conn)
        
        test_out_tbl = TimeseriesTable(**test_out)
        test_out_tbl.set_connection(conn)
        
        print('NOTE: Training set has {} observations'.format(train_out_tbl.shape[0]))
        print('NOTE: Validation set has {} observations'.format(valid_out_tbl.shape[0]))
        print('NOTE: Testing set has {} observations'.format(test_out_tbl.shape[0]))

        return  train_out_tbl, valid_out_tbl, test_out_tbl

    @staticmethod
    def generate_splitting_code(timeid, start, end, right_inclusive, 
                                partition_var_name, partition_val):
        if (start is None) and (end is not None):
            if right_inclusive:
                statement = '''if {0} <= {1} then {2} = '{3}';'''.format(
                        timeid, end, partition_var_name, partition_val)
            else:
                statement = '''if {0} < {1} then {2} = '{3}';'''.format(
                        timeid, end, partition_var_name, partition_val)
        elif (start is not None) and (end is None):
            statement = '''if {0} >= {1} then {2} = '{3}';'''.format(
                        timeid, start, partition_var_name, partition_val)
        elif (start is not None) and (end is not None):
            if right_inclusive:
                statement = '''if {0} >= {1} and {0} <= {2} then {3} = '{4}';'''.format(
                        timeid, start, end, partition_var_name, partition_val)
            else:
                statement = '''if {0} >= {1} and {0} < {2} then {3} = '{4}';'''.format(
                        timeid, start, end, partition_var_name, partition_val)
        else:
            statement = '''{0} = '{1}';'''.format(partition_var_name, partition_val)
            
        
        return statement

    @staticmethod
    def convert_to_sas_time_format(python_time, sas_format_type):
        if sas_format_type == 'date':
            if isinstance(python_time, datetime.date):
                sas_time_str = 'mdy({0},{1},{2})'.format(python_time.month,
                                   python_time.day, python_time.year)
                return sas_time_str
            elif python_time is None:
                return None
            else:
                raise ValueError('''The timeid type is date format, so the input 
                python time variable should be date or datetime format''')                
        elif sas_format_type == 'datetime':
            if isinstance(python_time, datetime.datetime):
                sas_time_str = 'dhms(mdy({0},{1},{2}), {3}, {4}, {5})'.format(
                        python_time.month, python_time.day, python_time.year, 
                        python_time.hour, python_time.minute, python_time.second)
                return sas_time_str
            elif isinstance(python_time, datetime.date):
                sas_time_str = 'dhms(mdy({0},{1},{2}), 0, 0, 0)'.format(
                        python_time.month, python_time.day, python_time.year)
                return sas_time_str
            elif python_time is None:
                return None
            else:
                raise ValueError('''The timeid type is datetime format, so the input 
                python time variable should be date or datetime format''')  
        elif sas_format_type == 'double':
            if isinstance(python_time, numbers.Real):
                return python_time
            elif python_time is None:
                return None
            else:
                raise ValueError('''The timeid type is double, so the input 
                python time variable should be int or float''') 
        else:
            raise DLPyError('''timeid format in CASTable is wrong, consider reload 
            the table and formatting it with timeseries_formatting''')
            
    @staticmethod
    def create_lags(varname, nlags, byvar):
        if not isinstance(byvar, list):
            byvar = [byvar]
        
        byvar_strlist = ['first.{}'.format(var) for var in byvar]            
       
        sascode = ''
        for i in range(nlags):
            if i == 0:
                sascode += '{0}_lag{1} = lag({0});'.format(varname, i+1)
            else:
                sascode += '{0}_lag{1} = lag({0}_lag{2});'.format(varname, i+1, i)
                
            if len(byvar) > 0:
                sascode += 'if ' + ' or '.join(byvar_strlist)
                sascode += ' then {0}_lag{1} = .;'.format(varname, i+1)
            
        return sascode     

    @staticmethod
    def find_file_caslib(conn, path):
        '''
        Check whether the specified path is in the caslibs of the current session
    
        Parameters
        ----------
        conn : CAS
            Specifies the CAS connection object
        path : string
            Specifies the name of the path.
    
        Returns
        -------
        ( flag, caslib_name )
            flag specifies if path exist in session.
            caslib_name specifies the name of the caslib that contains the path.
    
        '''
        paths = conn.caslibinfo().CASLibInfo.Path.tolist()
        caslibs = conn.caslibinfo().CASLibInfo.Name.tolist()
        subdirs = conn.caslibinfo().CASLibInfo.Subdirs.tolist()
    
        server_type = get_cas_host_type(conn).lower()
    
        if server_type.startswith("lin") or server_type.startswith("osx"):
            sep = '/'
        else:
            sep = '\\'
            
        for i, directory in enumerate(paths):
            if path.startswith(directory) and (subdirs[i]==1):
                rest_path = path[len(directory):]
                caslibname = caslibs[i]
                return (caslibname, rest_path)
            elif path.startswith(directory) and (subdirs[i]==0):
                rest_path = path[len(directory):]
                if sep in rest_path:
                    continue
                else:
                    caslibname = caslibs[i]
                    return (caslibname, rest_path)
        
        return (None, None)




def _get_first_obs(tbl, timeid, groupby=None, casout=None):
    
    input_tbl_name = tbl.name
    
    conn = tbl.get_connection()

    if casout is None:
        casout_params = {}
    elif isinstance(casout, CASTable):
        casout_params = casout.to_outtable_params()
    elif isinstance(casout, dict):
        casout_params = casout

    if 'name' not in casout_params:
        casout_params['name'] = random_name(input_tbl_name + '_first', 2)
        if len(casout_params['name']) >= 32:
            casout_params['name'] = random_name('tmp_first', 2)
        
    output_tbl_name = casout_params['name']

    if groupby is None:
        groupby = []
    elif not isinstance(groupby, list):
        groupby = [groupby]
    
    if not groupby:
        sascode = '''
        data {0};
        set {1};
        by {2};
        if _N_=1 then output {0};
        run;      
        '''.format(output_tbl_name, input_tbl_name, timeid, output_tbl_name)
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode, single='Yes')
    else:
        groupby_str = ' '.join(groupby)
        sascode = 'data {}; set {}; by {} {};'.format(output_tbl_name, 
                        input_tbl_name, groupby_str, timeid)
        
        condition_str = ['first.' + group for group in groupby]
        condition_str = ' or '.join(condition_str)
        sascode += 'if {} then output {};'.format(condition_str, output_tbl_name)
        sascode += 'run;'
        
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
        
    out = conn.CASTable(**casout_params)
    
    return out

def _get_last_obs(tbl, timeid, groupby=None, casout=None):
    
    input_tbl_name = tbl.name
    
    conn = tbl.get_connection()

    if casout is None:
        casout_params = {}
    elif isinstance(casout, CASTable):
        casout_params = casout.to_outtable_params()
    elif isinstance(casout, dict):
        casout_params = casout

    if 'name' not in casout_params:
        casout_params['name'] = random_name(input_tbl_name + '_last', 2)
        if len(casout_params['name']) >= 32:
            casout_params['name'] = random_name('tmp_last', 2)
        
    output_tbl_name = casout_params['name']

    if groupby is None:
        groupby = []
    elif not isinstance(groupby, list):
        groupby = [groupby]
    
    if not groupby:
        sascode = '''
        data {0};
        set {1} end=eof;
        by {2};
        if eof then output {0};
        run;      
        '''.format(output_tbl_name, input_tbl_name, timeid, output_tbl_name)
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode, single='Yes')
    else:
        groupby_str = ' '.join(groupby)
        sascode = 'data {}; set {}; by {} {};'.format(output_tbl_name, 
                        input_tbl_name, groupby_str, timeid)
        
        condition_str = ['last.' + group for group in groupby]
        condition_str = ' or '.join(condition_str)
        sascode += 'if {} then output {};'.format(condition_str, output_tbl_name)
        sascode += 'run;'
        
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
        
    out = conn.CASTable(**casout_params)
    
    return out

def _combine_table(tbl1=None, tbl2=None, columns=None, casout=None):
    
    conn = tbl1.get_connection()
    
    if casout is None:
        casout_params = {}
    elif isinstance(casout, CASTable):
        casout_params = casout.to_outtable_params()
    elif isinstance(casout, dict):
        casout_params = casout

    if 'name' not in casout_params:
        prefix = ''
        if tbl1 is not None:
            prefix += '_{}'.format(tbl1.name)
        
        if tbl2 is not None:
            prefix += '_{}'.format(tbl2.name)
            
        prefix = prefix.strip('_')
        casout_params['name'] = random_name(prefix, 1)
        if len(casout_params['name']) >= 32:
            casout_params['name'] = random_name('tmp_combine', 2)
        
        
    output_tbl_name = casout_params['name']
    
    if columns is None:
        keeps_str = ''
    else:
        if not isinstance(columns, list):
            columns = [columns]
        
        keeps_str = '(keep={})'.format(' '.join(columns))
    
    if tbl1 is None:
        sascode = '''
        data {};
        set {}{};
        run;
        '''.format(output_tbl_name, tbl2.name, keeps_str)       
    elif tbl2 is None:
        sascode = '''
        data {};
        set {}{};
        run;
        '''.format(output_tbl_name, tbl1.name, keeps_str)       
    else:       
        sascode = '''
        data {};
        set {}{} {}{};
        run;
        '''.format(output_tbl_name, tbl1.name, keeps_str, tbl2.name, keeps_str)
        
    conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)    
    out = conn.CASTable(**casout_params)
    
    return out
    

def _prepare_next_input(tbl, timeid, timeid_interval, autoregressive_series, 
                        sequence_opt, covar_tbl=None, groupby=None, casout=None):
    
    conn = tbl.get_connection()
    
    if casout is None:
        casout_params = {}
    elif isinstance(casout, CASTable):
        casout_params = casout.to_outtable_params()
    elif isinstance(casout, dict):
        casout_params = casout

    if 'name' not in casout_params:
        casout_params['name'] = random_name('next_input', 6)
        
    output_tbl_name = casout_params['name']
    
    if groupby is None:
        groupby = []
    elif not isinstance(groupby, list):
        groupby = [groupby]
    
    keeps_str = groupby + [timeid] + autoregressive_series[:-1]
    keeps_str += [sequence_opt['input_length'], sequence_opt['target_length']]
    keeps_str = ' '.join(keeps_str)
    
    assignment = []
    for i in range(len(autoregressive_series)-1):
        assignment += [autoregressive_series[i] + '=' + autoregressive_series[i+1]]
        
    assignment = ';'.join(assignment)
    
    sascode='''
    data {};
    set {};
    {} = intnx('{}', {}, 1);
    {};
    keep {};
    run;
    '''.format(output_tbl_name, tbl.name, timeid, timeid_interval, 
    timeid, assignment, keeps_str)

    conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
    
    if covar_tbl is not None:
        merge_by = groupby + [timeid]
        merge_by = ' '.join(merge_by)
        drops = autoregressive_series[:-1] + [sequence_opt['input_length'], sequence_opt['target_length']]
        drops = [var for var in drops if var in covar_tbl.columns.tolist()]
        drops = ' '.join(drops)
        sascode='''
        data {};
        merge {}(drop={} IN=in1) {}(IN=in2);
        by {};
        if in1=1 and in2=1 then output {};
        run;
        '''.format(output_tbl_name, covar_tbl.name, drops, output_tbl_name, 
        merge_by, output_tbl_name)
        
        conn.retrieve('dataStep.runCode', _messagelevel='error', code=sascode)
        
    out = conn.CASTable(**casout_params)
    
    return out
        
        
    
    



    

