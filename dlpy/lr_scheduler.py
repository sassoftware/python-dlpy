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

import math
from dlpy.utils import DLPyDict


class _LRScheduler(DLPyDict):
    """
    Learning rate scheduler

    Parameters
    ----------
    learning_rate_policy : string, optional
        Specifies the learning rate policy for the deep learning algorithm.
        Valid Values: FIXED, STEP, POLY, INV, MULTISTEP
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.
    power : double, optional
        Specifies the power for the learning rate policy.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches one
        of the specified steps, the learning rate is multiplied by the value
        of the gamma parameter. For example, if you specify {5, 9, 13}, then
        the learning rate is multiplied by gamma after the fifth, ninth, and
        thirteenth epochs.
    fcmp_learning_rate : string, optional
        specifies the FCMP learning rate function.

    Returns
    -------
    :class:`_LRScheduler`

    """
    def __init__(self, learning_rate_policy=None, learning_rate=None, gamma=None, steps=None, step_size=None,
                 power=None, fcmp_learning_rate=None):
        super(_LRScheduler, self).__init__(learning_rate_policy=learning_rate_policy, learning_rate=learning_rate,
                                           gamma=gamma, steps=steps, step_size=step_size, power=power,
                                           fcmp_learning_rate=fcmp_learning_rate)


class FCMPLR(_LRScheduler):
    """
    FCMP learning rate scheduler

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    fcmp_learning_rate : string
        specifies the FCMP learning rate function.
    learning_rate : double, optional
        Specifies the initial learning rate.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.

    Returns
    -------
    :class:`FCMPLR`

    """
    def __init__(self, conn, fcmp_learning_rate, learning_rate=0.001, gamma=0.1, step_size=10):
        if not conn.has_actionset('fcmpact'):
            conn.loadactionset(actionSet='fcmpact', _messagelevel='error')
        active_caslib_name = conn.caslibinfo(active=True).CASLibInfo.loc[0]['Name']
        active_caslib_name = 'CASUSER' if active_caslib_name.startswith('CASUSER(') else active_caslib_name
        conn.sessionProp.setsessopt(cmplib=active_caslib_name+'.'+fcmp_learning_rate)
        super(FCMPLR, self).__init__(fcmp_learning_rate=fcmp_learning_rate, learning_rate=learning_rate,
                                     gamma=gamma, step_size=step_size)


class FixedLR(_LRScheduler):
    """
    Fixed learning rate scheduler

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the learning rate for the deep learning algorithm.

    Returns
    -------
    :class:`FixedLR`

    """
    def __init__(self, learning_rate=0.001):
        _LRScheduler.__init__(self, learning_rate_policy='FIXED', learning_rate=learning_rate)


class StepLR(_LRScheduler):
    """
    Step learning rate scheduler

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the initial learning rate.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    step_size : int, optional
        Specifies the step size when the learning rate policy is set to STEP.

    Returns
    -------
    :class:`StepLR`

    """
    def __init__(self, learning_rate=0.001, gamma=0.1, step_size=10):
        _LRScheduler.__init__(self, learning_rate_policy='STEP', learning_rate=learning_rate,
                              gamma=gamma, step_size=step_size)


class MultiStepLR(_LRScheduler):
    """
    Multiple steps learning rate scheduler

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the initial learning rate.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    steps : list-of-ints, optional
        specifies a list of epoch counts. When the current epoch matches one
        of the specified steps, the learning rate is multiplied by the value
        of the gamma parameter. For example, if you specify {5, 9, 13}, then
        the learning rate is multiplied by gamma after the fifth, ninth, and
        thirteenth epochs.

    Returns
    -------
    :class:`MultiStepLR`

    """
    def __init__(self, learning_rate, gamma, steps):
        _LRScheduler.__init__(self, learning_rate_policy='MULTISTEP', learning_rate=learning_rate,
                              gamma=gamma, steps=steps)


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate scheduler

    Parameters
    ----------
    learning_rate : double, optional
        Specifies the initial learning rate.
    power : double, optional
        Specifies the power for the learning rate policy.

    Returns
    -------
    :class:`PolynomialLR`

    """
    def __init__(self, learning_rate, power):
        _LRScheduler.__init__(self, learning_rate_policy='POLY', learning_rate=learning_rate, power=power)


class ReduceLROnPlateau(FCMPLR):
    """
    Reduce learning rate on plateau learning rate scheduler

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    learning_rate : double, optional
        Specifies the initial learning rate.
    gamma : double, optional
        Specifies the gamma for the learning rate policy.
    cool_down_iters : int, optional
        Specifies number of iterations to wait before resuming normal operation after lr has been reduced.
    patience : int, optional
        Specifies number of epochs with no improvement after which learning rate will be reduced.

    Returns
    -------
    :class:`ReduceLROnPlateau`

    """
    def __init__(self, conn, learning_rate, gamma=0.1, cool_down_iters=10, patience=10):
        super(ReduceLROnPlateau, self).__init__(conn, learning_rate=learning_rate, gamma=gamma,
                                                fcmp_learning_rate='reduce_lr_on_plateau')
        conn.addRoutines(
            routineCode='''
                        function reduce_lr_on_plateau(rate, initRate, gamma, loss[*]);
                            len = dim(loss);
                            temp_rate = initRate;
                            cool_down_counter = {0};
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
                    
                                if bad_epoch > {1} then do;
                                    temp_rate = temp_rate * gamma;
                                    cool_down_counter = {0};
                                    bad_epoch = 0;
                                end;
                            end;
                            rate = temp_rate;
                            put rate=;
                            return(rate);
                        endsub;
                        '''.format(cool_down_iters, patience),
            package='pkg',
            funcTable=dict(name='reduce_lr_on_plateau', replace=1)
        )


class CyclicLR(FCMPLR):
    """
    Cyclic learning rate scheduler

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    data : string or CASTable
        Specifies the data for training.
    batch_size ： int
        Specifies the batch size equal to product of mini_batch_size, n_threads and number of workers.
    factor : int
        Specifies the number of epochs within one half of a cycle length
    learning_rate : double
        Specifies the initial learning rate that is smaller than max_lr.
    max_lr : double
        Specifies the highest learning rate.

    Returns
    -------
    :class:`CyclicLR`

    References
    ----------
    https://arxiv.org/pdf/1506.01186.pdf

    """
    def __init__(self, conn, data, batch_size, factor, learning_rate, max_lr):
        super(CyclicLR, self).__init__(conn, learning_rate=learning_rate,
                                       fcmp_learning_rate='cyclic_lr')
        num_batch_per_epoch = math.ceil(conn.numrows(data).numrows / batch_size)
        step_size = int(num_batch_per_epoch * factor)
        conn.addRoutines(
            routineCode='''
                        function cyclic_lr(rate, iterNum, batch, initRate);
                            batch_cum = {0} * iterNum + batch;
                            cycle = floor(batch_cum / (2 * {1}) + 1);
                            x = abs(batch_cum / {1} - 2 * cycle + 1);
                            rate = initRate + ({2} - initRate) * max(0, 1-x);
                            return(rate);
                        endsub;
                        '''.format(num_batch_per_epoch, step_size, max_lr),
            package='pkg',
            funcTable=dict(name='cyclic_lr', replace=1)
        )

