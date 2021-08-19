import os
from random import sample
import sys
try:
    from posix import listdir
except ModuleNotFoundError:
    from os import listdir

from utils import Session
import h5py
import numpy as np


def session2windows(in_dir, file, d, w, o, channels, temp_dim=False):
    try:
        sess = Session(file, in_dir)
    except:
        sys.exit("ERROR: Corrupted file")

    D = int(d * sess.SRATE)
    W = int(w * sess.SRATE)
    O = int(W * o)

    samples = []
    n_trials = sess.get_num_trials()
    for t in range(0,n_trials):
        trial_data = sess.cut_eeg(t, sess.SRATE)
        if(temp_dim):
            timestepsInBin = (D-O) // (W-O)
            n = ((trial_data.shape[1]-O)//(W-O))- timestepsInBin+1
        else:
            n = (trial_data.shape[1]-O) // (W - O)
        samples.append(n)

    samples = np.array(samples)
    samples[np.where(samples < 0)[0]] = 0

    if(temp_dim):
        data_shape = (sum(samples),timestepsInBin,channels,W)
    else:
        data_shape = (sum(samples),channels,W)

    data, labels = sess.get_bin_session(data_shape, W, O, D, temp_dim)

    return data, labels, samples


def dataset2windows(in_dir, out_dir, file_name, temp_dim, d, w, o, channels):
    files = os.listdir(in_dir)
    files.sort()
    
    
    for idx,f in enumerate(files):
        print("Processing " + f)
        
        data, labels, samples = session2windows(in_dir, f, d, w, o, channels, temp_dim)

        
        samples = np.cumsum(samples)[np.arange(0,450,25)[1:]]
        data = np.array(np.split(data,samples))
        labels = np.array(np.split(labels,samples))
        run_labels = list(np.array([1,1,2,1,1,3,1,1,4,1,1,2,1,1,3,1,1,4]).argsort())
        data = list(data[run_labels])
        labels = list(labels[run_labels])
        

        #init_shape = tuple(init_shape)
        hf = h5py.File(os.path.join(out_dir, file_name + "_" + os.path.splitext(f)[0] + '.h5'), 'w')
        
        hf.create_dataset('test_2d_data', data=np.concatenate(data[-2:]), compression="gzip")
        hf.create_dataset('test_2d_labels', data=np.concatenate(labels[-2:]), compression="gzip")
        del data[-2:], labels[-2:]
        hf.create_dataset('test_ud_data', data=np.concatenate(data[-2:]), compression="gzip")
        hf.create_dataset('test_ud_labels', data=np.concatenate(labels[-2:]), compression="gzip")
        del data[-2:], labels[-2:]
        hf.create_dataset('test_lr_data', data=np.concatenate(data[-2:]), compression="gzip")
        hf.create_dataset('test_lr_labels', data=np.concatenate(labels[-2:]), compression="gzip")
        del data[-2:], labels[-2:]
        hf.create_dataset('train_data', data=np.concatenate(data), compression="gzip")
        hf.create_dataset('train_labels', data=np.concatenate(labels), compression="gzip")
        del data, labels
        hf.close()


def usage():
    u = "Usage : python session2samples.py input_dir temp_dim\n\t \
        - input_dir: Directory where sessions are located\n\t \
        - output_dir: Directory where windowed sessions will be located\n\t \
        - file_name: The file name for the output files (ie:files will be named file_name_S1_Session1,file_name_S2_Session1 ...)\n\t\
        - temp_dim [Bool]: split data in temporal windows (1) or return it as an image (0)\n\t\
        - d [s]: Length of the bin in seconds. Only used if temp_dir is 1. If temp_dir = 0 then input d = 0\n\t \
        - w [s]: Length of the window in seconds \n\t \
        - o [%]: Overlaping percentage of windowing. The value must be in range [0,100) "
    return u


if __name__ == "__main__":
    if len(sys.argv) != 8:
        sys.exit(usage())
    try:
        temp_dim = int(sys.argv[4])
        if(temp_dim != 0 and temp_dim != 1):
            raise ValueError("temp_dim must be either 1 or 0")

        d = float(sys.argv[5])
        if(d < 0):
            raise ValueError("d must be positive")

        w = float(sys.argv[6])
        if(w < 0):
            raise ValueError("w must be positive")

        o = float(sys.argv[7])
        if(o < 0 or o >= 100):
            raise ValueError("d must be positive")
        o /= 100
    except:
        sys.exit(usage())

    if(not os.path.exists(sys.argv[1])):
        sys.exit("Input folder does not exist :(")

    if(not os.path.exists(sys.argv[2])):
        os.mkdir(sys.argv[2])

    n_channels = 62

    dataset2windows(sys.argv[1], sys.argv[2], sys.argv[3], temp_dim, d, w, o, n_channels)
