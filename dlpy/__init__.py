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

''' SAS Deep Learning '''

__version__ = '1.1.0'
__dev__ = False

from dlpy.model import Model
from .images import ImageTable
from .timeseries import TimeseriesTable
from dlpy.sequential import Sequential
from dlpy.layers import (Layer, InputLayer, Conv2d, Pooling, Dense, Recurrent, BN, Res, Concat, Proj, OutputLayer)
