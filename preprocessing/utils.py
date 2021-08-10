import pandas as pd
from scipy.io import loadmat
import numpy as np
import os

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
        self.SRATE = bci_data[4]                 # Sampling rate of EEG recording
        self.trialData = bci_data[5]             # Data structure describing trial level metrics
        self.metadata = bci_data[6]              # Participant and session level demographic information
        self.chaninfo = bci_data[7]              # Information about individual EEG channels

    def get_trial_data(self, trial_n):
        """
        Returns the trial information for a given trial:
        Parameters:
            - trial_n (int): trial number to return information for
        """
        
        taskNumber = self.trialData[0][trial_n][0]          # Identification number for task type
        runNumber = self.trialData[0][trial_n][1]           # The run to which a trial belongs
        trialNumber = self.trialData[0][trial_n][2]         # The trial number of a given session
        targetNumber = self.trialData[0][trial_n][3]        # Identification number for target presented (1=R, 2=L, 3=U, 4=D)
        trialLength = self.trialData[0][trial_n][4]         # The length of the feedback control period in s
        targetHitNumber = self.trialData[0][trial_n][5]     # Identification number for target selected by the BCI control
        resultInd = self.trialData[0][trial_n][6]           # Time index for the end of the feedback control portion of the trial
        result = self.trialData[0][trial_n][7]              # Outcome of the trial: success or failure
        forcedResult = self.trialData[0][trial_n][8]        # Outcome of the trial with forced target selection for timeout trials: success or failure
        artefact = self.trialData[0][trial_n][9]            # Does the trial contain an artefact?

        return taskNumber, runNumber, trialNumber, targetNumber, trialLength, targetHitNumber, resultInd, result, \
               forcedResult, artefact

    def cut_eeg(self, trial_n, pre=None, post=None):
        """
        Removes all data before/after pre/post values for a given trial.
        Parameters:
            - trial_n (int): trial number to split
            - pre (int): number of ms before target presentation to include in cut (includes bound)
            - post (int): number of ms after target presentation to include in cut (includes bound)
        Returns:
            - trial_cut (DataFrame): EEG data (channels x time)
        """
        _, _, _, _, triallength, _, _, _, _, _ = self.get_trial_data(trial_n)
        # Set our post_trial value to be the full trial length by default
        if post == None:
            post_trial = int((triallength[0][0]*1000))+2000
        # Set our pre_trial value to be the start of the feedback-control period (2000ms after 0)
        if pre == None:
            pre_trial = -2000

        trial_data = self.data[0][trial_n]
        trial_time = self.time[0][trial_n]
        dataframe = pd.DataFrame(trial_data, columns=trial_time[0])
        range_t = list(range(-pre_trial,post_trial+1))
        trial_cut = dataframe[range_t]

        return trial_cut


    def bin_trial(self, trial_cut, binlength=500, delay=40):
        """
        Splits a trial into multiple overlapping bins of EEG data.
        
        Parameters:
            - trial_cut (DataFrame) - A slice of EEG data (channel x time)
            - binlength (int) - Length of each bin (ms)
            - delay (int) - Delay between bins (ms)
        Returns:
            - bins (list) - A list containing EEG arrays (channels x time)
        """
        n_bins = ((trial_cut.shape[1]-binlength)//delay)+1         # Calculate the number of bins we can get from this trial
        bins = []
        start_t = 0                                            # Set a counter for which timepoint to start each bin on
        for i in range(1,n_bins+1):
            bin = trial_cut.iloc[:,start_t:start_t+binlength]
            bins.append(bin.to_numpy())
            start_t += delay                                   # Add our delay value to the counter

        return bins


    def get_x_y(self, pre=None, post=None):
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
        for trial in range(0,len(self.data[0])):
            if trial%50==0:
                print(f"Trial: {trial}/{len(self.data[0])}")
            _, _, _, targetnumber, _, _, _, _, _, _ = self.get_trial_data(trial)

            trial_cut = self.cut_eeg(trial, pre, post)      # Cut all the EEG data that we don't care about
            bins = self.bin_trial(trial_cut)                            # Split the trial into multiple bins
            label = np.repeat(targetnumber,len(bins))                   # Make an array of label numbers of the same size
            inputs.append(bins)
            labels.append(label)

        inputs = [input for input in inputs if input]           # Remove any empty arrays (caused by trials <500ms)
        inputs = np.concatenate(inputs,axis=0)                  # Convert our list of arrays into a single array
        labels = np.concatenate(labels,axis=0)
        return inputs, labels
