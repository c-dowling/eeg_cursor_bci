from scipy.io import loadmat
import os

class Session:
    def __init__(self, file, path):
        self.path = path
        self.file = file
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

        attributes:
            - trial_n (int): trial number to return information for
        """
        trial_data = self.mat_file["BCI"][0][0][5][0][trial_n]
        tasknumber = trial_data[0]        #
        runnumber = trial_data[1]
        trialnumber = trial_data[2]
        targetnumber = trial_data[3]
        triallength = trial_data[4]
        targethitnumber = trial_data[5]
        resultind = trial_data[6]
        result = trial_data[7]
        forcedresult = trial_data[8]
        artifact = trial_data[9]
        return tasknumber, runnumber, trialnumber, targetnumber, triallength, targethitnumber, resultind, result, \
               forcedresult, artifact

#    def get_x_y(self):

