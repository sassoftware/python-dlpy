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

# NOTE: This test requires a running CAS server.  You must use an ~/.authinfo
#       file to specify your username and password.  The CAS host and port must
#       be specified using the CASHOST and CASPORT environment variables.
#       A specific protocol ('cas', 'http', 'https', or 'auto') can be set using
#       the CASPROTOCOL environment variable.

import os

import swat
import datetime
import swat.utils.testing as tm
import numpy as np
import pandas as pd
import unittest
from dlpy.timeseries import TimeseriesTable
from dlpy.timeseries import plot_timeseries
from dlpy.utils import DLPyError


class TestTimeseriesTable(unittest.TestCase):
    # Create a class attribute to hold the cas host type
    server_type = None
    conn = None
    server_sep = '/'
    data_dir = None

    def setUp(self):
        swat.reset_option()
        swat.options.cas.print_messages = False
        swat.options.interactive_mode = False

        self.conn = swat.CAS()
        self.server_type = tm.get_cas_host_type(self.conn)
        self.server_sep = '\\'
        if self.server_type.startswith("lin") or self.server_type.startswith("osx"):
            self.server_sep = '/'

        if 'DLPY_DATA_DIR' in os.environ:
            self.data_dir = os.environ.get('DLPY_DATA_DIR')
            if self.data_dir.endswith(self.server_sep):
                self.data_dir = self.data_dir[:-1]
            self.data_dir += self.server_sep

        self.srcLib = tm.get_casout_lib(self.server_type)
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        filename2 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp2.txt')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        importoptions2 = dict(filetype='delimited', delimiter='\t')

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        filename3 = self.data_dir+'timeseries_exp1.csv'
        filename4 = self.data_dir+'timeseries_exp2.txt'

        self.table1 = TimeseriesTable.from_localfile(self.conn, filename1, importoptions=importoptions1)
        self.table2 = TimeseriesTable.from_localfile(self.conn, filename2, importoptions=importoptions2)
        self.table3 = TimeseriesTable.from_serverfile(self.conn, filename3, importoptions=importoptions1)
        self.table4 = TimeseriesTable.from_serverfile(self.conn, filename4, importoptions=importoptions2)
        self.table5 = TimeseriesTable.from_table(self.table1, columns=['id1var', 'date', 'series'])

        pandas_df1 = pd.read_csv(filename1)
        pandas_df2 = pandas_df1.series
        self.table6 = TimeseriesTable.from_pandas(self.conn, pandas_df1)
        self.table7 = TimeseriesTable.from_pandas(self.conn, pandas_df2)

        self.assertNotEqual(self.table1.name, None)
        self.assertNotEqual(self.table2.name, None)
        self.assertNotEqual(self.table3.name, None)
        self.assertNotEqual(self.table4.name, None)
        self.assertNotEqual(self.table5.name, None)
        self.assertNotEqual(self.table6.name, None)
        self.assertNotEqual(self.table7.name, None)

    def tearDown(self):
        # tear down tests
        try:
            self.conn.endsession()
        except swat.SWATError:
            pass
        del self.conn
        swat.reset_option()

    def test_table_loading2(self):
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')
        importoptions1 = dict(filetype='delimited', delimiter=',')
        importoptions2 = dict(filetype='delimited', delimiter='\t')

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        filename4 = self.data_dir+'timeseries_exp2.txt'

        table_tmp1 = TimeseriesTable.from_localfile(self.conn, filename1, importoptions=importoptions1,
                                                    casout=dict(name='table_tmp1'))
        table_tmp2 = TimeseriesTable.from_serverfile(self.conn, filename4, importoptions=importoptions2,
                                                     casout=dict(name='table_tmp2'))
        table_tmp3 = TimeseriesTable.from_table(self.table1, columns=['id1var', 'date', 'series'],
                                                casout=dict(name='table_tmp3'))

        pandas_df1 = pd.read_csv(filename1)
        pandas_df2 = pandas_df1.series
        table_tmp4 = TimeseriesTable.from_pandas(self.conn, pandas_df1, casout=dict(name='table_tmp4'))
        table_tmp5 = TimeseriesTable.from_pandas(self.conn, pandas_df2, casout=dict(name='table_tmp5'))

        self.assertEqual(table_tmp1.name, 'table_tmp1')
        self.assertEqual(table_tmp2.name, 'table_tmp2')
        self.assertEqual(table_tmp3.name, 'table_tmp3')
        self.assertEqual(table_tmp4.name, 'table_tmp4')
        self.assertEqual(table_tmp5.name, 'table_tmp5')

        tmp1_tbl = self.conn.CASTable('table_tmp1')
        localtmp1 = tmp1_tbl.to_frame()
        localtable1 = self.table1.to_frame()
        localtmp1 = localtmp1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        localtable1 = localtable1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        coltypes = {'id1var': np.float64, 'id2var': np.float64, 'covar': np.float64}
        localtmp1 = localtmp1.astype(coltypes)
        localtable1 = localtable1.astype(coltypes)
        self.assertTrue(localtable1.equals(localtmp1))

        tmp4_tbl = self.conn.CASTable('table_tmp4')
        localtmp4 = tmp4_tbl.to_frame()
        localtmp4 = localtmp4.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        pandas_df1 = pandas_df1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        coltypes = {'id1var':np.float64, 'id2var': np.float64, 'covar': np.float64}
        localtmp4 = localtmp4.astype(coltypes)
        pandas_df1 = pandas_df1.astype(coltypes)
        self.assertTrue(pandas_df1.equals(localtmp4))

    def test_table_loading3(self):
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')

        if self.data_dir is None:
            unittest.TestCase.skipTest(self, "DLPY_DATA_DIR is not set in the environment variables")

        filename3 = self.data_dir+'timeseries_exp1.csv'

        table_tmp1 = TimeseriesTable.from_localfile(self.conn, filename1, casout=self.conn.CASTable('table_tmp1'))
        table_tmp2 = TimeseriesTable.from_serverfile(self.conn, filename3,
                                                     casout=self.conn.CASTable('table_tmp1', replace=True))
        table_tmp3 = TimeseriesTable.from_table(self.table1, columns=['id1var', 'date', 'series'],
                                                casout=self.conn.CASTable('table_tmp2'))

        pandas_df1 = pd.read_csv(filename1)
        pandas_df2 = pandas_df1.series
        table_tmp4 = TimeseriesTable.from_pandas(self.conn, pandas_df1, casout=self.conn.CASTable('table_tmp4'))
        table_tmp5 = TimeseriesTable.from_pandas(self.conn, pandas_df2, casout=self.conn.CASTable('table_tmp5'))

        self.assertEqual(table_tmp1.name, 'table_tmp1')
        self.assertEqual(table_tmp2.name, 'table_tmp1')
        self.assertEqual(table_tmp3.name, 'table_tmp2')
        self.assertEqual(table_tmp4.name, 'table_tmp4')
        self.assertEqual(table_tmp5.name, 'table_tmp5')

        tmp1_tbl = self.conn.CASTable('table_tmp1')
        localtmp1 = tmp1_tbl.to_frame()
        localtable1 = self.table1.to_frame()
        localtmp1 = localtmp1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        localtable1 = localtable1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        coltypes = {'id1var': np.float64, 'id2var': np.float64, 'covar': np.float64}
        localtmp1 = localtmp1.astype(coltypes)
        localtable1 = localtable1.astype(coltypes)
        self.assertTrue(localtable1.equals(localtmp1))

        tmp4_tbl = self.conn.CASTable('table_tmp4')
        localtmp4 = tmp4_tbl.to_frame()
        localtmp4 = localtmp4.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        pandas_df1 = pandas_df1.sort_values(['id1var', 'id2var', 'datetime']).reset_index(drop=True)
        coltypes = {'id1var': np.float64, 'id2var': np.float64, 'covar': np.float64}
        localtmp4 = localtmp4.astype(coltypes)
        pandas_df1 = pandas_df1.astype(coltypes)
        self.assertTrue(pandas_df1.equals(localtmp4))

    def test_table_type(self):
        self.assertTrue(isinstance(self.table1, TimeseriesTable))
        self.assertTrue(isinstance(self.table2, TimeseriesTable))
        self.assertTrue(isinstance(self.table3, TimeseriesTable))
        self.assertTrue(isinstance(self.table4, TimeseriesTable))
        self.assertTrue(isinstance(self.table5, TimeseriesTable))
        self.assertTrue(isinstance(self.table6, TimeseriesTable))
        self.assertTrue(isinstance(self.table7, TimeseriesTable))

    def test_table_shape(self):
        self.assertEqual(self.table1.shape, (150,7))
        self.assertEqual(self.table2.shape, (150,7))
        self.assertEqual(self.table3.shape, (150,7))
        self.assertEqual(self.table4.shape, (150,7))
        self.assertEqual(self.table5.shape, (150,3))
        self.assertEqual(self.table6.shape, (150,7))
        self.assertEqual(self.table7.shape, (150,2))

    def test_table_formatting(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')
        tblinfo1 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo1.Format[tblinfo1.Column == 'datetime'].values[0], 'DATETIME')

        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.',
                                          timeid_format='DATE9.')
        tblinfo2 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo2.Format[tblinfo2.Column == 'date'].values[0], 'DATE')

        #check pandas input table
        self.table6.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')
        tblinfo3 = self.table6.columninfo().ColumnInfo
        self.assertEqual(tblinfo3.Format[tblinfo3.Column == 'datetime'].values[0], 'DATETIME')

        self.table6.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.',
                                          timeid_format='DATE9.')
        tblinfo4 = self.table6.columninfo().ColumnInfo
        self.assertEqual(tblinfo4.Format[tblinfo4.Column == 'date'].values[0], 'DATE')

    def test_table_formatting2(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.')
        tblinfo1 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo1.Format[tblinfo1.Column == 'datetime'].values[0], 'DATETIME')

        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.')
        tblinfo2 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo2.Format[tblinfo2.Column == 'date'].values[0], 'YYMMDD')

        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_format='DATE9.')
        tblinfo3 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo3.Format[tblinfo3.Column == 'date'].values[0], 'DATE')

    def test_table_formatting3(self):
        with self.assertRaises(ValueError):
            self.table1.timeseries_formatting(timeid='datetime',
                                              timeseries=['series', 'covar'])

    def test_table_formatting4(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series'],
                                          extra_columns='covar',
                                          timeid_informat='ANYDTDTM19.')

        tblinfo1 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo1.Format[tblinfo1.Column == 'datetime'].values[0], 'DATETIME')
        self.assertEqual(self.table1.shape, (150, 3))

    def test_table_formatting5(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series'],
                                          extra_columns='covar',
                                          timeid_informat='ANYDTDTM19.')

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series')

        tblinfo1 = self.table1.columninfo().ColumnInfo
        self.assertEqual(tblinfo1.Format[tblinfo1.Column == 'datetime'].values[0], 'DATETIME')
        self.assertEqual(self.table1.shape, (150, 3))

    def test_table_formatting6(self):
        with self.assertRaises(ValueError):
            self.table1.timeseries_formatting(timeid='datetime',
                                              timeseries=['series', 'failtest'],
                                              extra_columns='covar',
                                              timeid_informat='ANYDTDTM19.')

    def test_table_formatting7(self):
        with self.assertRaises(ValueError):
            self.table1.timeseries_formatting(timeid='datetime1',
                                              timeseries=['series'],
                                              extra_columns='covar',
                                              timeid_informat='ANYDTDTM19.')

    def test_table_accumulation(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        localtable = self.table1.to_frame()

        id1var_sel = np.random.choice(localtable.id1var.unique())
        id2var_sel = np.random.choice(localtable.id2var.unique())
        subgroup_tbl = localtable.loc[(localtable.id1var == id1var_sel) & (localtable.id2var==id2var_sel)]
        subgroup_tbl = subgroup_tbl.sort_values('datetime')

        self.assertEqual(subgroup_tbl.datetime.diff().min(), datetime.timedelta(days=1))
        self.assertEqual(subgroup_tbl.datetime.diff().max(), datetime.timedelta(days=1))

    def test_table_accumulation2(self):
        with self.assertRaises(DLPyError):
            self.table1.timeseries_accumlation(acc_interval='day',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation3(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        table8 = TimeseriesTable.from_table(self.table1)

        with self.assertRaises(DLPyError):
            table8.timeseries_accumlation(timeid='datetime',
                                          acc_interval='day',
                                          groupby=['id1var', 'id2var'])

    def test_table_accumulation4(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        table8 = TimeseriesTable.from_table(self.table1)
        self.assertFalse(hasattr(table8, 'timeseries'))

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        localtable1 = self.table1.to_frame()
        localseries1 = localtable1.series.sort_values()

        table8.timeseries_accumlation(timeid='datetime',
                                      timeseries = 'series',
                                      acc_interval='day',
                                      groupby=['id1var', 'id2var'])

        localtable8_1 = table8.to_frame()
        localseries8_1 = localtable8_1.series.sort_values()

        table8.timeseries_accumlation(timeid='datetime',
                                      timeseries = ['series'],
                                      acc_interval='day',
                                      groupby=['id1var', 'id2var'])

        localtable8_2 = table8.to_frame()
        localseries8_2 = localtable8_2.series.sort_values()

        self.assertTrue(localseries1.equals(localseries8_1))
        self.assertTrue(localseries8_2.equals(localseries8_1))

    def test_table_accumulation5(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='day',
                                               groupby=['id1var', 'id3var'])

    def test_table_accumulation6(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='day', timeid='wrong',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation7(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='day',
                                               timeseries = 'wrong',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation8(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='day',
                                               timeseries = 'wrong',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation9(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           extra_num_columns = 'covar',
                                           groupby=['id1var', 'id2var'])

    def test_table_accumulation10(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='day',
                                               timeseries = 'series',
                                               extra_num_columns = 'wrong',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation11(self):

        self.table1.timeseries_formatting(timeid='date',
                                          timeseries='series',
                                          timeid_informat='yymmdd10.')

        with self.assertRaises(ValueError):
            self.table1.timeseries_accumlation(acc_interval='minute',
                                               timeseries = 'series',
                                               groupby=['id1var', 'id2var'])

    def test_table_accumulation12(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           groupby=['id1var', 'id2var'],
                                           acc_method_byvar={'series':'avg'})

        localtbl1 = self.table1.to_frame()

        localtbl1 = localtbl1.sort_values(['id1var', 'id2var', 'datetime'])

        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.')

        self.table2.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           groupby=['id1var', 'id2var'],
                                           default_ts_acc='avg')

        localtbl2 = self.table2.to_frame()

        localtbl2 = localtbl2.sort_values(['id1var', 'id2var', 'datetime'])

        self.assertTrue(localtbl1.equals(localtbl2))

    def test_table_accumulation13(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           extra_num_columns = 'covar',
                                           groupby=['id1var', 'id2var'],
                                           acc_method_byvar={'covar':'avg'})

        localtbl1 = self.table1.to_frame()

        localtbl1 = localtbl1.sort_values(['id1var', 'id2var', 'datetime'])

        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.')

        self.table2.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           extra_num_columns = 'covar',
                                           groupby=['id1var', 'id2var'],
                                           default_col_acc='avg')

        localtbl2 = self.table2.to_frame()

        localtbl2 = localtbl2.sort_values(['id1var', 'id2var', 'datetime'])

        self.assertTrue(localtbl1.equals(localtbl2))

    def test_table_accumulation14(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries='series',
                                          timeid_informat='ANYDTDTM19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           timeseries = 'series',
                                           extra_num_columns = ['series','covar'],
                                           groupby=['id1var', 'id2var'],
                                           acc_method_byvar={'covar':'avg'})

    def test_table_subsequence(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        localtable = self.table1.to_frame()

        id1var_sel = np.random.choice(localtable.id1var.unique())
        id2var_sel = np.random.choice(localtable.id2var.unique())
        subgroup_tbl = localtable.loc[(localtable.id1var == id1var_sel) & (localtable.id2var==id2var_sel)]
        subgroup_tbl = subgroup_tbl.sort_values('datetime')

        start_time = datetime.datetime(2015, 1, 1) + datetime.timedelta(days=3)

        self.assertEqual(subgroup_tbl.datetime.min(), start_time)
        self.assertEqual(self.table1.shape, (105, 12))

    def test_table_subsequence2(self):

        # Test for different groupby inputs.
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        # table2
        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table2.timeseries_accumlation(acc_interval='day',
                                           extra_num_columns = 'id1var')

        self.table2.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         groupby='id1var',
                                         missing_handling='drop')

        # table3
        self.table3.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table3.timeseries_accumlation(acc_interval='day')

        with self.assertRaises(ValueError):
            self.table3.prepare_subsequences(seq_len=3,
                                             target='series',
                                             predictor_timeseries=['series', 'covar'],
                                             groupby = 'id3var',
                                             missing_handling='drop')

    def test_table_subsequence3(self):

        # Test for targets.
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        with self.assertRaises(DLPyError):
            self.table1.prepare_subsequences(seq_len=3,
                                             target=['series', 'covar'],
                                             predictor_timeseries=['series', 'covar'],
                                             missing_handling='drop')

    def test_table_subsequence4(self):

        # Test for predictor series.
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         missing_handling='drop')

        # Test for predictor series.
        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table2.timeseries_accumlation(acc_interval='day')

        self.table2.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries='covar',
                                         missing_handling='drop')

    def test_table_subsequence5(self):

        # Test for predictor series.
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         missing_handling='drop')

        # Test for predictor series.
        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table2.timeseries_accumlation(acc_interval='day')

        self.table2.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries='covar',
                                         missing_handling='drop')

    def test_table_subsequence6(self):

        # Test for error target.
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        with self.assertRaises(ValueError):
            self.table1.prepare_subsequences(seq_len=3,
                                             target='series1',
                                             missing_handling='drop')

        # Test for error independent variables.
        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table2.timeseries_accumlation(acc_interval='day')

        with self.assertRaises(ValueError):
            self.table2.prepare_subsequences(seq_len=3,
                                             target='series',
                                             predictor_timeseries='covar1',
                                             missing_handling='drop')

        # Test for sequence length
        self.table3.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table3.timeseries_accumlation(acc_interval='day')

        with self.assertRaises(ValueError):
            self.table3.prepare_subsequences(seq_len=0,
                                             target='series',
                                             predictor_timeseries='covar',
                                             missing_handling='drop')

    def test_table_subsequence7(self):

        # Test for timeseries warning
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['id2var'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           extra_num_columns=['series', 'covar'])

        self.table1.prepare_subsequences(seq_len=3,
                                         timeid='datetime',
                                         target='series',
                                         predictor_timeseries='covar',
                                         missing_handling='drop')

        self.table2.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table2.timeseries_accumlation(acc_interval='day')

        table8 = TimeseriesTable.from_table(self.table2)

        table8.prepare_subsequences(seq_len=3,
                                    timeid='datetime',
                                    target='series',
                                    predictor_timeseries='covar',
                                    missing_handling='drop')

    def test_table_subsequence8(self):

        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day')

        table8 = TimeseriesTable.from_table(self.table1)

        with self.assertRaises(ValueError):
            table8.prepare_subsequences(seq_len=3,
                                        target='series',
                                        predictor_timeseries='covar',
                                        missing_handling='drop')

        table9 = TimeseriesTable.from_table(self.table1)

        with self.assertRaises(ValueError):
            table9.prepare_subsequences(seq_len=3,
                                        timeid='wrong',
                                        target='series',
                                        predictor_timeseries='covar',
                                        missing_handling='drop')

    def test_table_partition(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = datetime.datetime(2015, 1, 7, 0 , 0, 0)
        test_start = datetime.date(2015, 1, 9)

        traintbl, validtbl, testtbl = self.table1.timeseries_partition(validation_start=valid_start,
                                                                       testing_start=test_start)

        self.assertTrue(isinstance(traintbl, TimeseriesTable))
        self.assertTrue(isinstance(validtbl, TimeseriesTable))
        self.assertTrue(isinstance(testtbl, TimeseriesTable))

        self.assertEqual(traintbl.name, self.table1.name + '_train')
        self.assertEqual(validtbl.name, self.table1.name + '_valid')
        self.assertEqual(testtbl.name, self.table1.name + '_test')

        localtrain = traintbl.to_frame()
        localtest = testtbl.to_frame()
        localvalid = validtbl.to_frame()

        self.assertEqual(localtrain.datetime.max(), datetime.datetime(2015, 1, 6, 0, 0, 0))
        self.assertEqual(localvalid.datetime.min(), datetime.datetime(2015, 1, 7, 0, 0, 0))
        self.assertEqual(localvalid.datetime.max(), datetime.datetime(2015, 1, 8, 0, 0, 0))
        self.assertEqual(localtest.datetime.min(), datetime.datetime(2015, 1, 9, 0, 0, 0))

        self.assertEqual(traintbl.inputs_target['inputs'], ['covar_lag2', 'series_lag3', 'covar_lag1', 'series_lag2',
                                                            'covar', 'series_lag1'])
        self.assertEqual(traintbl.inputs_target['target'], 'series')
        input_len_var = traintbl.sequence_opt['input_length']
        self.assertTrue(localtrain.loc[:,input_len_var].isin([3]).all())
        self.assertEqual(traintbl.sequence_opt['token_size'], 2)

    def test_table_partition2(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = datetime.date(2015, 1, 7)

        traintbl, validtbl, testtbl = self.table1.timeseries_partition(validation_start=valid_start)

        self.assertTrue(isinstance(traintbl, TimeseriesTable))
        self.assertTrue(isinstance(validtbl, TimeseriesTable))
        self.assertTrue(isinstance(testtbl, TimeseriesTable))

        self.assertEqual(traintbl.shape, (45, 13))
        self.assertEqual(validtbl.shape, (60, 13))
        self.assertEqual(testtbl.shape, (0, 13))

        localtrain = traintbl.to_frame()
        localtest = testtbl.to_frame()
        localvalid = validtbl.to_frame()

        self.assertEqual(localtrain.datetime.max(), datetime.datetime(2015, 1, 6, 0, 0, 0))
        self.assertEqual(localvalid.datetime.min(), datetime.datetime(2015, 1, 7, 0, 0, 0))
        self.assertEqual(localvalid.datetime.max(), datetime.datetime(2015, 1, 10, 0, 0, 0))

        self.assertEqual(traintbl.inputs_target['inputs'], ['covar_lag2', 'series_lag3', 'covar_lag1', 'series_lag2', 'covar', 'series_lag1'])
        self.assertEqual(traintbl.inputs_target['target'], 'series')
        input_len_var = traintbl.sequence_opt['input_length']
        self.assertTrue(localtrain.loc[:, input_len_var].isin([3]).all())
        self.assertEqual(traintbl.sequence_opt['token_size'], 2)

    def test_table_partition3(self):
        self.table1.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.',
                                          timeid_format='DATETIME19.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        traintbl, validtbl, testtbl = self.table1.timeseries_partition()

        self.assertTrue(isinstance(traintbl, TimeseriesTable))
        self.assertTrue(isinstance(validtbl, TimeseriesTable))
        self.assertTrue(isinstance(testtbl, TimeseriesTable))

        self.assertEqual(traintbl.shape, (105, 13))
        self.assertEqual(validtbl.shape, (0, 13))
        self.assertEqual(testtbl.shape, (0, 13))

        localtrain = traintbl.to_frame()
        localtest = testtbl.to_frame()
        localvalid = validtbl.to_frame()

        self.assertEqual(localtrain.datetime.min(), datetime.datetime(2015, 1, 4, 0, 0, 0))
        self.assertEqual(localtrain.datetime.max(), datetime.datetime(2015, 1, 10, 0, 0, 0))

        self.assertEqual(traintbl.inputs_target['inputs'], ['covar_lag2', 'series_lag3', 'covar_lag1',
                                                            'series_lag2', 'covar', 'series_lag1'])
        self.assertEqual(traintbl.inputs_target['target'], 'series')
        input_len_var = traintbl.sequence_opt['input_length']
        self.assertTrue(localtrain.loc[:,input_len_var].isin([3]).all())
        self.assertEqual(traintbl.sequence_opt['token_size'], 2)

    def test_table_partition4(self):
        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.',
                                          timeid_format='DATE9.')

        self.table1.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table1.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = datetime.date(2015, 1, 7)

        traintbl, validtbl, testtbl = self.table1.timeseries_partition(validation_start=valid_start)

        self.assertTrue(isinstance(traintbl, TimeseriesTable))
        self.assertTrue(isinstance(validtbl, TimeseriesTable))
        self.assertTrue(isinstance(testtbl, TimeseriesTable))

        self.assertEqual(traintbl.shape, (45, 13))
        self.assertEqual(validtbl.shape, (60, 13))
        self.assertEqual(testtbl.shape, (0, 13))

        localtrain = traintbl.to_frame()
        localtest = testtbl.to_frame()
        localvalid = validtbl.to_frame()

        self.assertEqual(localtrain.date.max(), datetime.date(2015, 1, 6))
        self.assertEqual(localvalid.date.min(), datetime.date(2015, 1, 7))
        self.assertEqual(localvalid.date.max(), datetime.date(2015, 1, 10))

        self.assertEqual(traintbl.inputs_target['inputs'], ['covar_lag2', 'series_lag3', 'covar_lag1',
                                                            'series_lag2', 'covar', 'series_lag1'])
        self.assertEqual(traintbl.inputs_target['target'], 'series')
        input_len_var = traintbl.sequence_opt['input_length']
        self.assertTrue(localtrain.loc[:,input_len_var].isin([3]).all())
        self.assertEqual(traintbl.sequence_opt['token_size'], 2)

    def test_table_partition5(self):
        self.table1.timeseries_formatting(timeid='id1var',
                                          timeseries=['series', 'covar'])

        self.table1.timeseries_accumlation(acc_interval='OBS',
                                           groupby=['id2var'])

        self.table1.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = 4

        traintbl, validtbl, testtbl = self.table1.timeseries_partition(validation_start=valid_start)

        self.assertTrue(isinstance(traintbl, TimeseriesTable))
        self.assertTrue(isinstance(validtbl, TimeseriesTable))
        self.assertTrue(isinstance(testtbl, TimeseriesTable))

        self.assertEqual(traintbl.shape, (6, 10))
        self.assertEqual(validtbl.shape, (3, 10))
        self.assertEqual(testtbl.shape, (0, 10))

        localtrain = traintbl.to_frame()
        localtest = testtbl.to_frame()
        localvalid = validtbl.to_frame()

        self.assertEqual(localtrain.id1var.min(), 2)
        self.assertEqual(localtrain.id1var.max(), 3)
        self.assertEqual(localvalid.id1var.min(), 4)
        self.assertEqual(localvalid.id1var.max(), 4)

        self.assertEqual(traintbl.inputs_target['inputs'], ['covar_lag1', 'series_lag2', 'covar', 'series_lag1'])
        self.assertEqual(traintbl.inputs_target['target'], 'series')
        input_len_var = traintbl.sequence_opt['input_length']
        self.assertTrue(localtrain.loc[:, input_len_var].isin([2]).all())
        self.assertEqual(traintbl.sequence_opt['token_size'], 2)

    def test_table_partition6(self):

        # Test for invalid time format for splitting.
        self.table1.timeseries_formatting(timeid='id1var', timeseries=['series', 'covar'])

        self.table1.timeseries_accumlation(acc_interval='OBS',
                                           groupby=['id2var'])

        self.table1.prepare_subsequences(seq_len=2,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = datetime.date(2015, 1, 7)

        with self.assertRaises(ValueError):
            traintbl, validtbl, testtbl = self.table1.timeseries_partition(validation_start=valid_start)

        # Test for invalid time format for splitting date.
        self.table2.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.')

        self.table2.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table2.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = 4

        with self.assertRaises(ValueError):
            traintbl, validtbl, testtbl = self.table2.timeseries_partition(validation_start=valid_start)

        # Test for invalid time format for splitting date.
        self.table3.timeseries_formatting(timeid='datetime',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='ANYDTDTM19.')

        self.table3.timeseries_accumlation(acc_interval='day',
                                           groupby=['id1var', 'id2var'])

        self.table3.prepare_subsequences(seq_len=3,
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = 4

        with self.assertRaises(ValueError):
            traintbl, validtbl, testtbl = self.table3.timeseries_partition(validation_start=valid_start)

        # Test for invalid time format for splitting date.
        self.table4.prepare_subsequences(seq_len=3, timeid='datetime',
                                         target='series',
                                         predictor_timeseries=['series', 'covar'],
                                         missing_handling='drop')

        valid_start = 'strings'

        with self.assertRaises(DLPyError):
            traintbl, validtbl, testtbl = self.table4.timeseries_partition(validation_start=valid_start)

    def test_table_plotting(self):
        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.')

        figure = plot_timeseries(self.table1, 'date', 'series',
                                 groupid=dict(id1var=0, id2var=1))

        figure2 = plot_timeseries(self.table1, 'date', 'series',
                                  groupid=dict(id1var=1, id2var=1), figure=figure,
                                  fontsize_spec={'xlabel': 16, 'ylabel': 16,
                                                 'xtick': 14, 'ytick': 14,
                                                 'legend': 14, 'title': 20})

        figure3 = plot_timeseries(self.table1, 'date', 'series',
                                  groupid=dict(id1var=2, id2var=1), figure=figure2,
                                  xlabel='xlab', ylabel='ylab', title='good',
                                  xlim=figure2[1].get_xlim(),
                                  ylim=figure2[1].get_ylim())

    def test_table_plotting2(self):
        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.')

        figure = plot_timeseries(self.table1, 'date', 'series',
                                 label='series', xdate_format='%d-%b-%Y',
                                 figsize=(12, 8),  xlabel='Datetime',
                                 ylabel='Temperature', fontsize_spec={'xlabel':40},
                                 title='great')

        figure2 = plot_timeseries(self.table1, 'date', 'series',
                                  groupid=dict(id1var=1, id2var=1), figure = figure,
                                  fontsize_spec={'xlabel': 16, 'ylabel': 16,
                                                 'xtick': 14, 'ytick': 14,
                                                 'legend': 14, 'title': 20})

    def test_table_plotting3(self):
        self.table1.timeseries_formatting(timeid='date',
                                          timeseries=['series', 'covar'],
                                          timeid_informat='yymmdd10.')

        figure = plot_timeseries(self.table1, 'date', 'series',
                                 start_time=datetime.date(2015, 1, 2),
                                 end_time = datetime.date(2015, 1, 8),
                                 groupid=dict(id1var=1, id2var=1))

        self.table2.timeseries_formatting(timeid='id1var', timeseries=['series', 'covar'])

        figure2 = plot_timeseries(self.table2, 'id1var', 'series',
                                  start_time=1,
                                  end_time =3,
                                  groupid=dict(id2var=1))

    def test_table_plotting4(self):
        filename1 = os.path.join(os.path.dirname(__file__), 'datasources', 'timeseries_exp1.csv')

        pandas_df1 = pd.read_csv(filename1)
        pandas_df1['date'] = pd.to_datetime(pandas_df1.date, format='%Y-%m-%d')
        pandas_df2 = pandas_df1.series

        figure = plot_timeseries(pandas_df1, 'date', 'series', groupid=dict(id1var=1, id2var=1))

        figure2 = plot_timeseries(pandas_df2, 'date', 'series')
