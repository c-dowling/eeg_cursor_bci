import os, sys
from posix import listdir
from utils import Session
import h5py


def session2windows(in_dir, file, d, w, o, data, labels, temp_dim = False):
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
            data, labels = sess.bin_trial(trial_data, t, data, labels, D, W, O, temp_dim)
        else:
            #TODO
            raise NotImplementedError("Functionality not available :(")

    return data, labels



def dataset2windows(in_dir, out_dir, file_name, temp_dim, d, w, o):
    files = os.listdir(in_dir)
    files.sort()
    
    
    for f in files:
        print("Processing "+f)
        data = []
        labels = []
        data, labels = session2windows(in_dir, f, d, w, o, data, labels, temp_dim)
        # https://pytorch.org/docs/0.3.0/data.html#torch.utils.data.ConcatDataset
        hf = h5py.File(os.path.join(out_dir,file_name+"_"+os.path.splitext(f)[0]+'.h5'), 'w')
        hf.create_dataset('data', data=data)
        hf.create_dataset('label', data=labels)
        hf.close()



if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit("Usage : python session2samples.py input_dir temp_dim\n\t \
        - input_dir: Directory where sessions are located\n\t \
        - output_dir: Directory where windowed sessions will be located\n\t \
        - file_name: The file name for the output files (ie:files will be named file_name_S1_Session1,file_name_S2_Session1 ...)\
        - temp_dim: split data in temporal windows (1) or return it as an image (0)")
    try:
        temp_dim = int(sys.argv[4])
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
    dataset2windows(sys.argv[1], sys.argv[2], sys.argv[3], temp_dim, d, w, o)