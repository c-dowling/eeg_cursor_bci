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
        self.data = f.load('data').value
        self.labels = f.load('labels').value

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.FloatTensor(self.data[idx, :, :]).unsqueeze(0)
        label = torch.LongTensor(self.labels[idx]).squeeze()

        return data, label


class CustomTemporalDataset(Dataset):
    def __init__(self, file_name):
        try:
            f = h5py.File(file_name, 'r')
        except FileNotFoundError:
            sys.exit("Unable to open {}".format(file_name))
        self.data = f['data']
        self.labels = f['labels']

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.FloatTensor(self.data[idx, :, :]).unsqueeze(1)
        label = torch.LongTensor(self.labels[idx]).squeeze()

        return data, label


def concat_datasets(input_dir):
    datasets = []
    file_names = os.listdir(input_dir)
    file_names.sort()
    for f in file_names:
        if (f.endswith('.h5')):
            datasets.append(CustomDataset(os.path.join(input_dir, f)))
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
