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

''' TensorBoard class provides functionality for viewing scalar metrics in TensorBoard '''

import os
import swat

class TensorBoard():
    '''
    Parameters
    ----------
    model : dlpy.Model
        Specifies the desired model object to monitor.
    log_dir : string
        Specifies the directory to write logs to.
    use_valid : bool
        Specifies whether to record validation statistics.
        If set to True then user must pass valid_table to Model.fit().
        Default: False

    Returns
    -------
    :class:`TensorBoard`

    '''
    def __init__(self, model, log_dir, use_valid=False):
        self.model = model
        if os.path.exists(log_dir):
            self.log_dir = log_dir
        else:
            raise OSError(log_dir + " does not exist. Please provide an existing directory to write event logs or create this directory.")
        self.use_valid = use_valid

        # Scalar metrics to log
        self.scalars = ['learning_rate', 'loss', 'error']
        if self.use_valid:
            self.scalars.append('valid_' + self.scalars[1])
            self.scalars.append('valid_' + self.scalars[2])

    def build_summary_writer(self):
        '''
        Creates a SummaryWriter object for logging scalar events 
        to the appropriate directory. Will return a dictionary of writers
        with elements for each scalar to be monitored.

        Returns
        -------
        dictionary
            dictionary where keys are individual scalars and values are the 
            SummaryWriter for a scalar.
        '''
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow must be installed to use tensorboard")

        writer_dict = {}

        for i in self.scalars:
            writer_dict[i] =  tf.summary.create_file_writer(
                self.log_dir + self.model.model_table['name'] + '/' + i + '/'
            )
                
        return writer_dict

    def log_scalar(self, summary_writer, scalar_name, scalar_value, scalar_global_step):
        '''
        Writes a scalar summary value as a tfevents file.

        Parameters
        ----------
        summary_writer : SummaryWriter
            The SummaryWriter object for a particular scalar (e.g. the SummaryWriter for learning rate).
        scalar_name : string
            The scalar that is being recorded (e.g. learning rate).
        scalar_value : np.float64
            A substring of the CASResponse message that contains the scalar value to log
        scalar_global_step : str
            A substring of the CASResponse message that contains the step (iteration) value of the scalar so far
            (e.g. "10" would be the 10th iteration of a certain scalar, by default using the epoch count).
        '''
        try:
            import tensorflow as tf
        except ImportError:
            print("TensorFlow must be installed to use tensorboard")
            
        with summary_writer.as_default():
            tf.summary.scalar(name=scalar_name, data=scalar_value, step=scalar_global_step)
            summary_writer.flush()

    def tensorboard_response_cb(self, response, connection, userdata):
        '''
        Callback function to handle the CASResponse message while model training is run. 
        This function is called after each iteration of Model.fit() in order to capture 
        the scalar training metrics that are being monitored. This function calls log_scalar() 
        to write the needed tfevents files for tensorboard.

        Parameters
        ----------
        response : swat.cas.response.CASResponse
            The CASResponse from a CAS action. 
        connection : swat.cas.connection.CAS
            The CAS connection object
        userdata : swat.cas.results.CASResults() or arbitrary user data structure
            Keeps state information between calls. The returned value of the function
            will get passed in as the userdata argument on the next call.

        Returns
        -------
        swat.cas.results.CASResults()
            Each time function is called it returns whatever is stored in userdata
            and is passed into subsequent function calls as the userdata parameter.
        '''
        # Initialize userdata as a CASResults instance to hold results of action 
        # Initialize userdata attribute, message, to store the response message from action,
        # at_scalar to determine when to log, writer_dict to hold our summary writers for each
        # scalar, and epoch_count to keep track of correct epoch value for model.
        if userdata is None:
            userdata = swat.cas.results.CASResults()
            userdata.message = None
            userdata.at_scaler = False
            userdata.severity = 0
            userdata.writer_dict = self.build_summary_writer()
            userdata.epoch_count = self.model.n_epochs + 1
            
        # Store the CASResults in userdata
        for k,v in response:
            userdata[k] = v

        # Update userdata severity 
        userdata.severity = response.disposition.severity

        # Get the initial response message
        if userdata.message is None:
            userdata.message = response.messages
            
        # Skip if reponse is an empty list
        if not userdata.message:
            pass
        
        # Writing scalar data
        else:
            # Split current message
            userdata.message = userdata.message[0].split()

            # Change at_scaler flag when done training
            if 'optimization' in userdata.message:
                userdata.at_scaler = False

            # Wait until next epoch
            if 'Batch' in userdata.message:
                userdata.at_scaler = False
            
            # Log scalers at each epoch
            if userdata.at_scaler:
                for k,v in userdata.writer_dict.items():
                    if k == 'learning_rate':
                        self.log_scalar(v, k, float(userdata.message[2]), userdata.epoch_count)
                    if k == 'loss':
                        self.log_scalar(v, k, float(userdata.message[3]), userdata.epoch_count)
                    if k == 'error':
                        self.log_scalar(v, k, float(userdata.message[4]), userdata.epoch_count)
                    if k == 'valid_loss':
                        self.log_scalar(v, k, float(userdata.message[5]), userdata.epoch_count)
                    if k == 'valid_error':
                        self.log_scalar(v, k, float(userdata.message[6]), userdata.epoch_count)
                        
                # Increment the epoch_count 
                userdata.epoch_count += 1
                
            # Change at_scaler flag if we are ready to log
            if 'Epoch' in userdata.message:
                userdata.at_scaler = True

            # Get next response
            userdata.message = response.messages

        return userdata
