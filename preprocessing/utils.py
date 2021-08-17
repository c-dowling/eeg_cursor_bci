import os
import sys

from scipy.io import loadmat


class Session:
    """Provides EEG data, session information and trial information from a session.mat file."""

    def __init__(self, file, path):
        """Initialises a Session object.

        Params:
            - path (str): The path where file is stored (e.g. "../data")
            - file (str): filename (e.g. "S1_Session_1.mat")
        """
        self.path = path
        self.file = file
        self.datadir = os.path.join(path, file)
        mat_file = loadmat(self.datadir)
        self.get_bci_data(mat_file)

    def get_bci_data(self, mat_file):
        """Stores session information to the Session object.

        Params:
            - mat_file (dict): A dictionary consisting of a loaded .mat file.
        """
        bci_data = mat_file["BCI"][0][0]
        self.data = bci_data[0]                  # EEG data from each trial of the session
        self.time = bci_data[1]                  # Vector of the trial time (in ms) relative to target presentation
        self.positionx = bci_data[2]             # X position cursor during feedback
        self.positiony = bci_data[3]             # Y position of cursor during feedback
        self.SRATE = int(bci_data[4])            # Sampling rate of EEG recording
        self.trialData = bci_data[5][0]          # Data structure describing trial level metrics
        self.metadata = bci_data[6]              # Participant and session level demographic information
        self.chaninfo = bci_data[7]              # Information about individual EEG channels

    def get_num_trials(self):
        """Returns the number of trials in a given session

        Returns:
            - len(self.data[0]) (int): Number of trials in a given session.
        """
        return len(self.data[0])

    def get_trial_length(self, trial_n):
        """Returns the length of a given trial (samples)

        Params:
            - trial_n (int): Trial number
        Returns:
            - self.trialData[trial_n][4] (int): Length of a trial (samples)
        """
        return self.trialData[trial_n][4]

    def get_target_num(self, trial_n):
        """Returns the label indicating the position of the target on a particular tria

        Params:
        - trial_n (int): Trial number
        """
        return self.trialData[trial_n][3]

    def get_trial_data(self, trial_n):
        """
        Returns the trial information for a given trial:

        Params:
            - trial_n (int): Trial number to return information for
        Returns:
            - taskNumber (int): Identification number for task type
            - runNumber (int): The run to which a trial belongs
            - trialNumber (int): The trial number of a given session
            - targetNumber (int): Identification number for target presented (1=R, 2=L, 3=U, 4=D)
            - trialLength (float): The length of the feedback control period in s
            - targetHitNumber (int): Identification number for target selected by the BCI control
            - resultInd (int): Time index for the end of the feedback control portion of the trial
            - result (int): Outcome of the trial: success or failure
            - forcedResult (int): Outcome of the trial with forced target selection for timeout trials: success or failure
            - artefact (int): Does the trial contain an artefact?
        """
        
        taskNumber = self.trialData[trial_n][0].item()
        runNumber = self.trialData[trial_n][1].item()
        trialNumber = self.trialData[trial_n][2].item()
        targetNumber = self.trialData[trial_n][3].item()
        trialLength = self.trialData[trial_n][4].item()
        targetHitNumber = self.trialData[trial_n][5].item()
        resultInd = self.trialData[trial_n][6].item()
        result = self.trialData[trial_n][7].item()
        forcedResult = self.trialData[trial_n][8].item()
        artefact = self.trialData[trial_n][9].item()

        return taskNumber, runNumber, trialNumber, targetNumber, trialLength, targetHitNumber, resultInd, result, \
               forcedResult, artefact

    def cut_eeg(self, trial_n, sr=None, start_t=None, end_t=None):
        """
        Removes all data before/after pre/post values for a given trial.

        Params:
            - trial_n (int): trial number to split
            - sr (int): sample rate
            - start_t (int): number of ms before target presentation to include in cut (includes bound)
            - end_t (int): number of ms after target presentation to include in cut (includes bound)
        Returns:
            - trial_cut (array): EEG data (channels x time)
        """
        if sr is None:
            sr = self.SRATE

        # Set our post_trial value to be the full trial length by default
        if end_t is None:
            end_t = int(-1 * sr)

        # Set our pre_trial value to be the start of the feedback-control period (2000ms after 0)
        if start_t is None:
            start_t = 4 * sr + 1

        trial_data = self.data[0][trial_n]
        trial_cut = trial_data[:, start_t:end_t]

        return trial_cut

    def bin_trial(self, trial_cut, num_trial, data, labels, window, overlap, bin=500, temp_dim=0, save=True):
        """
        Splits a trial into multiple windows of EEG data.

        Params:
            - trial_cut (array) - A slice of EEG data (channels x samples)
            - num_trial (int) - The trial number
            - data (list) - A list where the input data will be appended
            - labels (list) - A list where the data labels will be appended
            - window (int) - Length of each window (samples)
            - overlap (int) - Samples that overlap between windows (samples)
            - temp_dim (int) - Whether to use temporal windowing or not
            - save (bool) - Whether to save data or not
        Returns:
            - data (list) - A list containing the arrays for each EEG window (channels x time)
            - labels (list) - A list containing the labels which correspond to each EEG window
        """
        if(temp_dim == 0):
            n_bins = (trial_cut.shape[1] - overlap) // (window - overlap)         # Calculate the number of bins we can get from this trial
            # n_bins = (trial_cut.shape[1]-(binlength-delay))//delay         # Calculate the number of bins we can get from this trial
            for i in range(0, n_bins):
                data.append(trial_cut[:, i * (window - overlap): i * (window - overlap) + window])
                labels.append(self.get_target_num(num_trial))
        else:
            if(bin%(window-overlap)!=0):
                sys.exit("Bins of {} samples cannot be divided into windows of {} samples with {} overlapping samples".format(bin,window,overlap))
            n_timesteps = (trial_cut.shape[1]-overlap)//(window-overlap)
            timestepsInBin = (bin-overlap) // (window-overlap)
            for i in range (0,n_timesteps-timestepsInBin+1):
                windows = []
                for t in range(0, timestepsInBin):
                    windows.append(trial_cut[:, (i + t) * (window - overlap):(i + t) * (window - overlap) + window])
                data.append(windows)
                labels.append(self.get_target_num(num_trial))

        
            
        return data, labels
            #raise NotImplementedError("Function not available yet :(")    # Add our delay value to the counter


        return data, labels

    def get_inputs_labels(self, start_t=None, end_t=None, binlength=500, delay=40, overlap=0, temp_dim=0):
        """
        Returns two arrays containing the input data for the model and corresponding labels.
        Params:
            - pre (int): If True will set number of ms before target presentation to include in cut (includes bound)
            - post (int): If True will set number of ms after target presentation to include in cut (includes bound)
            - start_t (int): number of ms before target presentation to include in cut (includes bound)
            - end_t (int): number of ms after target presentation to include in cut (includes bound)
            - binlength (int) - Length of each window (samples)
            - delay (int) - Delay between bins (samples)
            - overlap (int) - _____________________
            - temp_dim (int) - _____________________
        Returns:
            - inputs (array): A 3D array of EEG data for each window (window x channels x time)
            - labels (array): A 1D array of intended cursor direction for each bin
        """
        inputs = []
        labels = []
        for trial in range(0, len(self.data[0]) - 1):
            if trial % 50 == 0:
                print(f"Extracting Trial Data: {trial}/{len(self.data[0])}")

            trial_cut = self.cut_eeg(trial, start_t, end_t)
            inputs, labels = self.bin_trial(trial_cut, trial, inputs, labels, binlength, delay, overlap, temp_dim, save=False)

        return inputs, labels
