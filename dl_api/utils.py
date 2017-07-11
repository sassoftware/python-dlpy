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

import random
import string


def random_name(name='ImageData', length=6):
    '''
    Function to generate ramdom name.
    '''

    return name + '_' + ''.join(random.sample(
        string.ascii_uppercase + string.ascii_lowercase + string.digits, length))


def input_table_check(input_table):
    '''
    Function to unify the input_table format.


    Parameters:

    ----------

    input_table : A CAS table object, a string specifies the name of the CAS table,
                a dictionary specifies the CAS table, or an Image object.

    Return:

    ----------
    A dictionary specifies the CAS table

    '''
    type_indicator = input_table.__class__.__name__
    if type_indicator == "str":
        input_table = dict(name=input_table)
    elif type_indicator == "dict":
        input_table = input_table
    elif type_indicator == "Image":
        input_table = input_table.tbl
    elif type_indicator == "CASTable":
        input_table = dict(name=input_table.tableinfo().TableInfo.Name[0])
    else:
        raise TypeError('input_table must be one of the following:\n'
                        '1. A CAS table object;\n'
                        '2. A string specifies the name of the CAS table,\n'
                        '3. A dictionary specifies the CAS table\n'
                        '4. An Image object.')
    return input_table
