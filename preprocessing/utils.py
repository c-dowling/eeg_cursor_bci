import pandas as pd
from scipy.io import loadmat
import numpy as np
import os
import h5py

class Session:
    def __init__(self, file, path):
        self.path = path                    # path where file is stored (e.g. "../data")
        self.file = file                    # filename (e.g. "S1_Session_1.mat")
        self.datadir = os.path.join(path,file)
        mat_file = loadmat(self.datadir)
        self.get_bci_data(mat_file)

    def get_bci_data(self, mat_file):
        bci_data = mat_file["BCI"][0][0]
        self.data = bci_data[0]                  # EEG data from each trial of the session
        self.time = bci_data[1]                  # Vector of the trial time (in ms) relative to target presentation
        self.positionx = bci_data[2]             # X position cursor during feedback
        self.positiony = bci_data[3]             # Y position of cursor during feedback
        self.SRATE = int(bci_data[4])                 # Sampling rate of EEG recording
        self.trialData = bci_data[5][0]             # Data structure describing trial level metrics
        self.metadata = bci_data[6]              # Participant and session level demographic information
        self.chaninfo = bci_data[7]              # Information about individual EEG channels

    def get_num_trials(self):
        return len(self.data[0])

    def get_trial_length(self, trial_n):
        return self.trialData[trial_n][4]

    def get_target_num(self, trial_n):
        return self.trialData[trial_n][3]

    def get_trial_data(self, trial_n):
        """
        Returns the trial information for a given trial:
        Parameters:
            - trial_n (int): trial number to return information for
        """
        
        taskNumber = self.trialData[trial_n][0]          # Identification number for task type
        runNumber = self.trialData[trial_n][1]           # The run to which a trial belongs
        trialNumber = self.trialData[trial_n][2]         # The trial number of a given session
        targetNumber = self.trialData[trial_n][3]        # Identification number for target presented (1=R, 2=L, 3=U, 4=D)
        trialLength = self.trialData[trial_n][4]         # The length of the feedback control period in s
        targetHitNumber = self.trialData[trial_n][5]     # Identification number for target selected by the BCI control
        resultInd = self.trialData[trial_n][6]           # Time index for the end of the feedback control portion of the trial
        result = self.trialData[trial_n][7]              # Outcome of the trial: success or failure
        forcedResult = self.trialData[trial_n][8]        # Outcome of the trial with forced target selection for timeout trials: success or failure
        artefact = self.trialData[trial_n][9]            # Does the trial contain an artefact?

        return taskNumber, runNumber, trialNumber, targetNumber, trialLength, targetHitNumber, resultInd, result, \
               forcedResult, artefact

    def cut_eeg(self, trial_n, sr=None, start_t=None, end_t=None):
        """
        Removes all data before/after pre/post values for a given trial.
        Parameters:
            - trial_n (int): trial number to split
            - sr (int): sample rate
            - pre (int): number of ms before target presentation to include in cut (includes bound)
            - post (int): number of ms after target presentation to include in cut (includes bound)
        Returns:
            - trial_cut (DataFrame): EEG data (channels x time)
        """
        if sr == None:
            sr = self.SRATE

        # Set our post_trial value to be the full trial length by default
        if end_t == None:
            end_t = int(-1 * sr)

        # Set our pre_trial value to be the start of the feedback-control period (2000ms after 0)
        if start_t == None:
            start_t = 4*sr+1

        trial_data = self.data[0][trial_n]
        trial_cut = trial_data[:, start_t:end_t]

        return trial_cut


    def bin_trial(self, trial_cut, num_trial, data, labels, binlength=500, delay=40, overlap=0, temp_dim=0, save=True):
        """
        Splits a trial into multiple overlapping bins of EEG data.
        
        Parameters:
            - trial_cut (DataFrame) - A slice of EEG data (channel x time)
            - binlength (int) - Length of each bin (samples)
            - delay (int) - Delay between bins (samples)
        Returns:
            - bins (list) - A list containing EEG arrays (channels x time)
        """
        if(temp_dim==0):
            #print(trial_cut.shape[1])
            n_bins = (trial_cut.shape[1]-(binlength-delay))//delay         # Calculate the number of bins we can get from this trial                                           # Set a counter for which timepoint to start each bin on
            for i in range(0,n_bins):
                data.append(trial_cut[:, i*delay : i*delay+binlength])
                labels.append(self.get_target_num(num_trial))
            return data, labels
        else:
            raise NotImplementedError("Function not available yet :(")    # Add our delay value to the counter


    def get_inputs_labels(self, pre=None, post=None):
        """
        Returns two arrays containing the input data for the model and corresponding labels.
        Parameters:
            - pre (int): If True will set number of ms before target presentation to include in cut (includes bound)
            - post (int): If True will set number of ms after target presentation to include in cut (includes bound)
        Returns:
            - inputs (array): A 3D array of EEG data for each bin (bin x channels x time)
            - labels (array): A 1D array of intended cursor direction for each bin
        """
        inputs = []         # Create a blank list to store our input data arrays
        labels = []         # Create another list to store the label arrays
        for trial in range(0,len(self.data[0])-1):
            if trial%50==0:
                print(f"Extracting Trial Data: {trial}/{len(self.data[0])}")

            trial_cut = self.cut_eeg(trial, pre, post)
            inputs, labels = self.bin_trial(trial_cut, trial, inputs, labels)

        return inputs, labels
