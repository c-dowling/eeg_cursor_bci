import sys
import os

import torch
import h5pickle as h5py
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, file_name):
        try:
            f = h5py.File(file_name, 'r')
        except FileNotFoundError:
            sys.exit("Unable to open {}".format(file_name))
        self.data = f.get('data')[()]
        self.labels = f.get('labels')[()]-1


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.FloatTensor(self.data[idx, :, :]).unsqueeze(0)
        label = torch.LongTensor(self.labels[idx]).squeeze()

        return data, label


class CustomTemporalDataset(Dataset):
    def __init__(self, file_name, split=False, state=None):
        try:
            f = h5py.File(file_name, 'r')
        except FileNotFoundError:
            sys.exit("Unable to open {}".format(file_name))

        if split:
            try:
                self.data = f.get(state+"_data")[()]
                self.labels = f.get(state+"_labels")[()] - 1
            except ValueError("ERROR: Unacceptable value given for 'state'."):
                sys.exit()
        else:
            self.data = f.get('data')[()]
            self.labels = f.get('labels')[()] - 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        data = torch.FloatTensor(self.data[idx]).unsqueeze(1)
        label = torch.LongTensor([self.labels[idx]]).squeeze()

        return data, label

class CustomSplitTemporalDataset():
    def __init__(self, file):
        self.train = CustomTemporalDataset(file, split=True,state="train")
        self.test_lr = CustomTemporalDataset(file, split=True,state="test_lr")
        self.test_ud = CustomTemporalDataset(file, split=True,state="test_ud")
        self.test_2d = CustomTemporalDataset(file, split=True,state="test_2d")

def concat_datasets(input_dir, isTemporal = False):
    datasets = []
    file_names = os.listdir(input_dir)
    file_names.sort()
    for f in file_names:
        if (f.endswith('.h5')):
            if isTemporal:
                datasets.append(CustomTemporalDataset(os.path.join(input_dir,f)))
            else:
                datasets.append(CustomDataset(os.path.join(input_dir,f)))

    datasets = torch.utils.data.ConcatDataset(datasets)
    return datasets


class SequentialSampler(torch.utils.data.Sampler):
    """Sample sequentially (only for validation and test)"""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)
