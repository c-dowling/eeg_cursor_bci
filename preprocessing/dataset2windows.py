import os, sys
from posix import listdir
from utils import Session


def session2windows(in_dir, out_dir, file, d, w, o, temp_dim = False):
    try:
        sess = Session(file, in_dir)
    except:
        sys.exit("ERROR: Corrupted file")
    D = int(d*sess.SRATE)
    W = int(w*sess.SRATE)
    O = int(W*o)
    n_trials = sess.get_num_trials()
    for t in range(0,n_trials):
        trial_data = sess.cut_eeg(t, sess.SRATE)
        if(not temp_dim):
            #TODO
            file = os.path.splitext(file)[0]
            sess.bin_trial(trial_data, t, os.path.join(out_dir, file), sess.SRATE, D, W, O, temp_dim)
            #raise NotImplementedError("Functionality not available :(")
            
        else:
            #TODO
            raise NotImplementedError("Functionality not available :(")



def dataset2windows(in_dir, out_dir, temp_dir, d, w, o):
    files = os.listdir(in_dir)
    files.sort()
    '''
    for f in files:
        session2windows(in_dir, out_dir, f, d, w, o, temp_dir)
    '''
    session2windows(in_dir, out_dir, files[0], d, w, o, temp_dir)
    



if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("Usage : python session2samples.py input_dir temp_dim\n\t \
        - input_dir: Directory where sessions are located\n\t \
        - temp_dim: split data in temporal windows (1) or return it as an image (0)")
    try:
        temp_dim = int(sys.argv[3])
        if(temp_dim!=0 and temp_dim!=1):
            raise ValueError("temp_dim must be either 1 or 0")
    except:
        sys.exit()
    d = 500*1e-3
    w = 40*1e-3
    o = 0*1e-3
    if(not os.path.exists(sys.argv[1])):
        sys.exit("Input folder does not exist :(")
    if(not os.path.exists(sys.argv[2])):
        os.mkdir(sys.argv[2])
    dataset2windows(sys.argv[1], sys.argv[2], temp_dim, d, w, o)