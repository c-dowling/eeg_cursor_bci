import pandas as pd
import numpy as np
from scipy.io import loadmat
import os

class Session:
    def __init__(self, file, path):
        self.path = path                    # path where file is stored (e.g. "../data")
        self.file = file                    # filename (e.g. "S1_Session_1.mat")
        self.datadir = os.path.join(path,file)
        self.mat_file = loadmat(self.datadir)

    def get_bci_data(self):
        bci_data = self.mat_file["BCI"][0][0]
        data = bci_data[0]                  # EEG data from each trial of the session
        time = bci_data[1]                  # Vector of the trial time (in ms) relative to target presentation
        positionx = bci_data[2]             # X position cursor during feedback
        positiony = bci_data[3]             # Y position of cursor during feedback
        SRATE = bci_data[4]                 # Sampling rate of EEG recording
        TrialData = bci_data[5]             # Data structure describing trial level metrics
        metadata = bci_data[6]              # Participant and session level demographic information
        chaninfo = bci_data[7]              # Information about individual EEG channels

        return data, time, positionx, positiony, SRATE, TrialData, metadata, chaninfo

    def get_trial_data(self, trial_n):
        """
        Returns the trial information for a given trial:

        Parameters:
            - trial_n (int): trial number to return information for
        """
        trial_data = self.mat_file["BCI"][0][0][5][0][trial_n]
        tasknumber = trial_data[0]          # Identification number for task type
        runnumber = trial_data[1]           # The run to which a trial belongs
        trialnumber = trial_data[2]         # The trial number of a given session
        targetnumber = trial_data[3]        # Identification number for target presented (1=R, 2=L, 3=U, 4=D)
        triallength = trial_data[4]         # The length of the feedback control period in s
        targethitnumber = trial_data[5]     # Identification number for target selected by the BCI control
        resultind = trial_data[6]           # Time index for the end of the feedback control portion of the trial
        result = trial_data[7]              # Outcome of the trial: success or failure
        forcedresult = trial_data[8]        # Outcome of the trial with forced target selection for timeout trials: success or failure
        artefact = trial_data[9]            # Does the trial contain an artefact?
        return tasknumber, runnumber, trialnumber, targetnumber, triallength, targethitnumber, resultind, result, \
               forcedresult, artefact


    def cut_eeg(self, trial_n, pre, post):
        """
        Removes all EEG data before/after pre/post values.

        Parameters:
            - trial_n (int): trial number to split
            - pre (int): number of ms before target presentation to include in cut (includes bound)
            - post (int): number of ms after target presentation to include in cut (includes bound)
        """
        data, time, positionx, positiony, SRATE, TrialData, metadata, chaninfo = self.get_bci_data()
        trial_data = data[0][trial_n]
        trial_time = time[0][trial_n]
        dataframe = pd.DataFrame(trial_data, columns=trial_time[0])
        cols_to_keep = list(range(-pre,post+1))
        data_cut = dataframe[cols_to_keep]

        return data_cut


#    def bin_trial(self, binlength, overlap):
        """
        Splits a trial into multiple overlapping bins of EEG data.
        
        Parameters:
        """

#    def get_x_y(self):
        """
        Returns labelled input data as
        """
