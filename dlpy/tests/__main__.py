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

import argparse
import sys
import os.path
import unittest


def get_username():
    ''' Returns username '''
    if os.name != 'nt':
        import pwd #only on linux python
        return pwd.getpwuid(os.getuid())[0]
    else:
        return os.getlogin()


def run_py_tests(test_module, test_prefix='test_', to_console=True):
    print('...Current PYTHONPATH Being USED...')
    for each in sys.path:        
        print(each)
    
    XUNIT=False
    try:
        import xmlrunner
        XUNIT=True
    except ImportError:
        XUNIT=False
        print('...unable to location xmlrunner package')
        print('...output will not be in Xunit format')
    
    print('...running all tests in module %s that start with %s ' % (test_module, test_prefix))

    loader = unittest.TestLoader()
    loader.testMethodPrefix = test_prefix
    suite = loader.discover(test_module)         
              
    if to_console:            
        if XUNIT:
            xmlrunner.XMLTestRunner().run(suite)
        else:
            unittest.TextTestRunner().run(suite)
    else:#log file
        if XUNIT:
            xmlrunner.XMLTestRunner(output='dlpy-tests.log').run(suite)
        else:
            reportFile = open('dlpy-tests.log','w')
            unittest.TextTestRunner(stream=reportFile).run(suite)
            reportFile.close()  


defaults = {
    'host': os.environ.get('CASHOST', 'localhost'),
    'port': int(os.environ.get('CASPORT', 5570)),
    'user': os.environ.get('CASUSER', get_username()),
    'password': os.environ.get('CASPASSWORD', ''),
    'protocol': os.environ.get('CASPROTOCOL', 'auto'),
    'test_module': 'dlpy.tests',
    'test_prefix': 'test_',
}

parser = argparse.ArgumentParser(add_help=True)

parser.add_argument('--host', default=defaults['host'], help='name of server host (default is %s)' % defaults['host'])
parser.add_argument('--port', default=defaults['port'], type=int, help='server host port (default is %s)' % defaults['port'])
parser.add_argument('--user', default=defaults['user'], help='user name connecting to host')
parser.add_argument('--protocol', default=defaults['protocol'], help='protocol of CAS communications')
parser.add_argument('--test-module', default=defaults['test_module'],
                    help='module to test (default is %s)' % defaults['test_module'])
parser.add_argument('--test-prefix', default=defaults['test_prefix'],
                    help='prefix to discover tests within module(default is %s)' % defaults['test_prefix'])
parser.add_argument('--to-console', action='store_true', default=True,
                    help='display output to console, else will be logged as file')

args = parser.parse_args()

os.environ['CASHOST'] = args.host
os.environ['CASPORT'] = str(args.port)
os.environ['CASUSER'] = args.user
if args.protocol != 'auto':
    os.environ['CASPROTOCOL'] = args.protocol

if args.port:
    run_py_tests(test_module=args.test_module, test_prefix=args.test_prefix, to_console=args.to_console)
else:
    print('...did not provide a port to your host via --port, use --help for options')
    sys.exit(1)
