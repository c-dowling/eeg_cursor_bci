import os
import sys
try:
    from posix import listdir
except ModuleNotFoundError:
    from os import listdir

from utils import Session
import h5py


def session2windows(in_dir, file, d, w, o, data, labels, temp_dim=False):
    try:
        sess = Session(file, in_dir)
    except:
        sys.exit("ERROR: Corrupted file")
    D = int(d * sess.SRATE)
    W = int(w * sess.SRATE)
    O = int(W * o)
    n_trials = sess.get_num_trials()

    for t in range(0, n_trials):
        trial_data = sess.cut_eeg(t, sess.SRATE)
        file = os.path.splitext(file)[0]
        data, labels = sess.bin_trial(trial_data, t, data, labels, W, O, D, temp_dim)

    return data, labels


def session2windows_ttype(in_dir, file, d, w, o, train_data, train_labels, lr_data, lr_labels, ud_data, ud_labels,
                          twod_data, twod_labels, temp_dim=False):
    try:
        sess = Session(file, in_dir)
    except:
        sys.exit("ERROR: Corrupted file")
    D = int(d * sess.SRATE)
    W = int(w * sess.SRATE)
    O = int(W * o)

    for i in range(6):
        for j in range(50):
            t = j+i*75
            trial_data = sess.cut_eeg(t, sess.SRATE)
            file = os.path.splitext(file)[0]
            train_data, train_labels = sess.bin_trial(trial_data, t, train_data, train_labels, W, O, D, temp_dim)

    for i in range(2):
        for j in range(50,75):
            t = j+i*225
            trial_data = sess.cut_eeg(t, sess.SRATE)
            file = os.path.splitext(file)[0]
            lr_data, lr_labels = sess.bin_trial(trial_data, t, lr_data, lr_labels, W, O, D, temp_dim)

    for i in range(2):
        for j in range(125,150):
            t = j+i*225
            trial_data = sess.cut_eeg(t, sess.SRATE)
            file = os.path.splitext(file)[0]
            ud_data, ud_labels = sess.bin_trial(trial_data, t, ud_data, ud_labels, W, O, D, temp_dim)

    for i in range(2):
        for j in range(200,225):
            t = j+i*225
            trial_data = sess.cut_eeg(t, sess.SRATE)
            file = os.path.splitext(file)[0]
            twod_data, twod_labels = sess.bin_trial(trial_data, t, twod_data, twod_labels, W, O, D, temp_dim)

    return train_data, train_labels, lr_data, lr_labels, ud_data, ud_labels, twod_data, twod_labels


def dataset2windows(in_dir, out_dir, file_name, temp_dim, d, w, o, split_by_ttype=False):
    files = os.listdir(in_dir)
    files.sort()

    for f in files:
        print("Processing " + f)
        if split_by_ttype:
            train_data = []
            train_labels = []
            lr_data = []
            lr_labels = []
            ud_data = []
            ud_labels = []
            twod_data = []
            twod_labels = []
            train_data, train_labels, lr_data, lr_labels, ud_data, ud_labels, twod_data, twod_labels = \
            session2windows_ttype(in_dir, f, d, w, o, train_data, train_labels, lr_data, lr_labels, ud_data, ud_labels,
                                  twod_data, twod_labels, temp_dim)

            hf = h5py.File(os.path.join(out_dir, file_name + "_" + os.path.splitext(f)[0] + '.h5'), 'w')
            hf.create_dataset('train_data', data=train_data, compression="gzip", compression_opts=9)
            hf.create_dataset('train_labels', data=train_labels, compression="gzip", compression_opts=9)
            hf.create_dataset('lr_data', data=lr_data, compression="gzip", compression_opts=9)
            hf.create_dataset('lr_labels', data=lr_labels, compression="gzip", compression_opts=9)
            hf.create_dataset('ud_data', data=ud_data, compression="gzip", compression_opts=9)
            hf.create_dataset('ud_labels', data=ud_labels, compression="gzip", compression_opts=9)
            hf.create_dataset('twod_data', data=twod_data, compression="gzip", compression_opts=9)
            hf.create_dataset('twod_labels', data=twod_labels, compression="gzip", compression_opts=9)
            hf.close()

        else:
            data = []
            labels = []
            data, labels = session2windows(in_dir, f, d, w, o, data, labels, temp_dim)

            hf = h5py.File(os.path.join(out_dir, file_name + "_" + os.path.splitext(f)[0] + '.h5'), 'w')
            hf.create_dataset('data', data=data, compression="gzip", compression_opts=9)
            hf.create_dataset('labels', data=labels, compression="gzip", compression_opts=9)
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

    dataset2windows(sys.argv[1], sys.argv[2], sys.argv[3], temp_dim, d, w, o, split_by_ttype=True)
