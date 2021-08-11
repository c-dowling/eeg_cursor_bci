import torch
import numpy as np
from scipy.io import loadmat


class BCIDataset(torch.utils.data.Dataset):
    """
    Loads the BCI dataset. It loads trimmed signals from a .mat file.
    Update when the new dataset is ready.
    """
    def __init__(self, root, transforms, min_size=500):
        super().__init__()
        self.data = loadmat(root)
        self.signals = []
        self.labels = []
        self.transforms = transforms
        self.min_size = min_size
        self._process_data()

    def _process_data(self):
        signals = self.data['datasetEEG'][0][0][0][0]
        labels = self.data['datasetEEG'][0][0][1][0]

        for i_trial, (inputs, labels) in enumerate(zip(signals, labels)):
            if inputs.shape[1] > self.min_size:
                self.signals.append(inputs)
                self.labels.append(labels)

        self.labels = torch.tensor(self.labels).long() - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]

        signal = torch.tensor(signal)
        signal = self.transforms(signal)
        signal = signal.unsqueeze(0)

        return signal, label


class SequentialSampler(torch.utils.data.Sampler):
    """Sample sequentially (only for validation and test)"""
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (index for index in self.indices)

    def __len__(self):
        return len(self.indices)


def load_bci_dataset(root, split_size, batch_size, transforms):
    """Loads dataloaders.

    Args:
        root (str): Path to datset.
        split_size (tuple): Proportion train/valid/test e.g, (0.8, 0.1, 0.1)
        batch_size (int): Number of batches per epoch.
        transforms (torchvision.transforms.Compose): Group of transformation to apply to data.

    Returns:
        dict: Dictionary with trainloader, validloader and testloader.
    """
    assert sum(split_size) == 1

    dataset = BCIDataset(root=root, transforms=transforms)

    total_size = len(dataset)
    train_size = int(split_size[0] * total_size)
    valid_size = int(split_size[1] * total_size)
    test_size = total_size - train_size - valid_size

    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_idx = indices[valid_size + test_size:]
    valid_idx = indices[:valid_size]
    test_idx = indices[valid_size:valid_size + test_size]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)
    test_sampler = SequentialSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    dataloaders = {'Train': trainloader, 'Valid': validloader, 'Test': testloader}

    print(f'Num. Trials Train: [{len(train_idx)}/{total_size}]  Valid: [{len(valid_idx)}/{total_size}]  Test: [{len(test_idx)}/{total_size}]')

    return dataloaders
