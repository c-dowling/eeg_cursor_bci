import os, sys
from posix import listdir
from utils import Session


def session2windows(path, file, d, w, o, temporalDimension = False):
    sess = Session(file, path)
    D = d*sess.SRATE
    W = w*sess.SRATE
    O = int(W*o)
    n_trials = sess.get_num_trials()
    for t in range(1,n_trials+1):
        trial_data = sess.cut_eeg(t)
        if(not temporalDimension):
            #TODO
            sess.bin_trial(trial_data, bin_length)
            raise NotImplementedError("Functionality not available :(")
            
        else:
            #TODO
            raise NotImplementedError("Functionality not available :(")



def dataset2windows(path, temp_dir, d, w, o):
    files = os.listdir(path)
    for f in files:
        session2windows(path, f, d, w, o, temp_dir)



if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage : python session2samples.py input_dir temp_dim\n\t \
        - input_dir: Directory where sessions are located\n\t \
        - temp_dim: split data in temporal windows (1) or return it as an image (0)")
    try:
        temp_dim = int(sys.argv[2])
        if(temp_dim!=0 and temp_dim!=1):
            raise ValueError("temp_dim must be either 1 or 0")
    except:
        sys.exit()
    d = 500
    w = 40
    o = 0
    dataset2windows(sys.argv[1], temp_dim, d, w, o)