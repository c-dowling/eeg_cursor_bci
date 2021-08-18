import os
import torch
import h5pickle as h5py
from torch.utils.data import Dataset, ConcatDataset, Sampler


class CustomDataset(Dataset):
    def __init__(self, file: str, is_temporal: bool, state: str):
        """
        Dataset to read data from the file
        args:
        file: .h5 file location
        is_temporal: was data stored in temporal form
        state: what state do you want to read
        """
        self.unsqueeze_index = 1 if is_temporal else 0
        try:
            f = h5py.File(file, 'r')
            self.data = f.get(state + "_data")[()]
            self.labels = f.get(state + "_labels")[()] - 1

        except (FileNotFoundError, ValueError) as e:
            raise(e)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = torch.FloatTensor(self.data[idx, :, :]).unsqueeze(self.unsqueeze_index)
        label = torch.LongTensor(self.labels[idx]).squeeze()

        return data, label

def concat_datasets(input_dir: str, is_temporal: bool, states=None):
    if states is None:
        states = ['lr', 'train', 'twod', 'ud']
    """
    function to concatenate datasets from different files
    args:
    input_dir: directory where .h5 files are kept
    is_temporal: did you use temporal windows
    states: states from our dataset
    """
    # Create a dictionary for every state in our dataset
    datasets = {state: [] for state in states}
    file_names = os.listdir(input_dir)
    # Iterate through all the files and read the data
    for f in file_names:
        if f.endswith('.h5'):
            for state in states:
                datasets[state].append(CustomDataset(os.path.join(input_dir, f), is_temporal, state))

    # Create ConcatDataset from every state
    for state in states:
        datasets[state] = ConcatDataset(datasets[state])

    return datasets


class SequentialSampler(Sampler):
    """Sample sequentially (only for validation and test)"""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)
