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

from .model import DLPyDict
from .utils import random_name


class _LRScheduler(DLPyDict):
    def __init__(self, learning_rate_policy='FIXED', learning_rate=0.001, gamma=None, steps=None, step_size=None,
                 power=0.75, fcmp_learning_rate=None):
        DLPyDict.__init__(self, learning_rate_policy=learning_rate_policy, learning_rate=learning_rate, gamma=gamma,
                          steps=steps, step_size=step_size, power=power, fcmp_learning_rate=fcmp_learning_rate)


class FCMPLR(_LRScheduler):

    def __init__(self, conn, fcmp_learning_rate, learning_rate=0.001, gamma=0.1, step_size=10):
        if not conn.has_actionset('fcmp'):
            conn.loadactionset(actionSet = 'fcmp', _messagelevel = 'error')
        self.lr_function = random_name(self.__class__.__name__, 6)
        active_caslib_name = conn.caslibinfo(active = True).CASLibInfo.loc[0]['Name']
        active_caslib_name = 'CASUSER' if active_caslib_name.startswith('CASUSER(') else active_caslib_name
        conn.sessionProp.setsessopt(cmplib = f'{active_caslib_name}.{self.lr_function}')
        _LRScheduler.__init__(self, fcmp_learning_rate=fcmp_learning_rate, learning_rate = learning_rate, gamma=gamma,
                              step_size=step_size)


class FixedLR(_LRScheduler):

    def __init__(self, learning_rate=0.001):
        _LRScheduler.__init__(self, learning_rate_policy='FIXED', learning_rate=learning_rate)


class StepLR(_LRScheduler):

    def __init__(self, learning_rate=0.001, gamma=0.1, step_size=10):
        _LRScheduler.__init__(self, learning_rate_policy='STEP', learning_rate=learning_rate, gamma=gamma,
                              step_size=step_size)


class MultiStepLR(_LRScheduler):

    def __init__(self, learning_rate, gamma, steps):
        _LRScheduler.__init__(self, learning_rate_policy='MULTISTEP', learning_rate=learning_rate, gamma=gamma,
                              steps=steps)


class PolynomialLR(_LRScheduler):

    def __init__(self, learning_rate, power):
        _LRScheduler.__init__(self, learning_rate_policy='POLY', learning_rate=learning_rate, power=power)


class ReduceLROnPlateau(FCMPLR):

    def __init__(self, conn, learning_rate, gamma=0.1, cool_down_iters=10, patience=10):
        super(ReduceLROnPlateau, self).__init__(conn, learning_rate = learning_rate, gamma = gamma,
                                                fcmp_learning_rate = 'reduce_lr_on_plateau')
        conn.addRoutines(
            routineCode = f'''
                        function reduce_lr_on_plateau(rate, initRate, gamma, loss[*]);
                            len = dim(loss);
                            temp_rate = initRate;
                            cool_down_counter = {cool_down_iters};
                            best = loss[1];
                            do i=1 to len;
                    
                                if loss[i] < best then do;
                                    best = loss[i];
                                    bad_epoch = 0;
                                end;
                                else bad_epoch = bad_epoch + 1;
                    
                                if cool_down_counter > 0 then do;
                                    cool_down_counter = cool_down_counter - 1;
                                    bad_epoch = 0;
                                end;
                    
                                if bad_epoch > {patience} then do;
                                    temp_rate = temp_rate * gamma;
                                    cool_down_counter = {cool_down_iters};
                                    bad_epoch = 0;
                                end;
                            end;
                            rate = temp_rate;
                            put rate=;
                            return(rate);
                        endsub;
                        ''',
            package = 'pkg',
            funcTable = dict(name = self.lr_function, replace = 1)
        )
        del self.__dict__['lr_function']


class CyclicLR(FCMPLR):

    def __init__(self, conn, learning_rate, max_lr, step_size):
        super(CyclicLR, self).__init__(conn, learning_rate=learning_rate, step_size=step_size,
                                       fcmp_learning_rate='cyclic_lr')
        conn.addRoutines(
            routineCode = f'''
                        function cyclic_lr(rate, batch, initRate, stepSize);
                            cycle = floor(batch / (2 * stepSize) + 1);
                            x = abs(batch / stepSize - 2 * cycle + 1);
                            rate = initRate + ({max_lr} - initRate) * max(0, 1-x);
                            return(rate);
                        endsub;
                        ''',
            package = 'pkg',
            funcTable = dict(name = self.lr_function, replace = 1)
        )
        del self.__dict__['lr_function']

