from utils import Session

file = "S1_Session_1.mat"
path = "../data"

a = Session(file,path)

tasknumber, runnumber, trialnumber, targetnumber, triallength, targethitnumber, resultind, result, \
forcedresult, artifact = a.get_trial_data(trial_n=34)
print(triallength)
