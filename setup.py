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

''' Install the SAS Deep Learning package '''

import io
import os
from setuptools import setup, find_packages


def get_file(fname):
    with io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), fname),
                 encoding='utf8') as infile:
        return infile.read()

setup(
    name='sas-dlpy',
    version='1.1.0',
    description='SAS Deep Learning Interface',
    long_description=get_file('README.md'),
    long_description_content_type='text/markdown',
    author='SAS',
    author_email='support@sas.com',
    url='https://github.com/sassoftware/python-dlpy/',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
        'dlpy': ['datasources/*', 'tests/datasources/*'],
    },
    #include_package_data=True,
    install_requires=[
        'swat >= 1.5.0',
        'pandas >= 0.16.0',
        'six >= 1.9.0',
        'graphviz',
        'matplotlib',
        'pillow'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
    ],
)
