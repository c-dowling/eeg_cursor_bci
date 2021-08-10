import numpy as np
from utils import Session

file = "S1_Session_1.mat"
path = "../../data"

# Initialise a Session object
a = Session(file,path)

# Get the trial data for trial number 25
tasknumber, runnumber, trialnumber, targetnumber, triallength, targethitnumber, resultind, result, \
forcedresult, artefact = a.get_trial_data(trial_n=25)

# Print the length of the feedback control period (in seconds) for trial 25
print(f"End of feedback control (ms): {resultind[0][0]}")

# Get the EEG data for trial 25 (feedback control period only)
datacut = a.cut_eeg(25, -2000, int(triallength[0][0]*1000))
print(datacut)


# We can call the below function to get an array containing each of our inputs and labels
inputs, labels = a.get_x_y()


# n_trials = 450 - (n_trials < 500ms)
# n_bins per trial = variable
# n_channels = 62
# len_bin = 500ms

# Our dimensions should be N (n_trials x n_bins_per_trial), 62 (n_channels), 500 (length of bins)
print(f"Dimensions of input and labels: {inputs.shape,labels.shape}")

