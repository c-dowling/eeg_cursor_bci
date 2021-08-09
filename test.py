from utils import Session

file = "S1_Session_1.mat"
path = "../data"

a = Session(file,path)

# Get the trial data for trial number 25
tasknumber, runnumber, trialnumber, targetnumber, triallength, targethitnumber, resultind, result, \
forcedresult, artifact = a.get_trial_data(trial_n=25)

# Print the length of the feedback control period (in seconds) for trial 25
print(triallength)

# Get the EEG data for trial 25 (20ms before stim onset - 10 seconds after stim onset)
datacut = a.cut_eeg(25, 20, 10)
print(datacut)