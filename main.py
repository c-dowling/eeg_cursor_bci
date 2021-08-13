from preprocessing.dataloaders import concat_datasets, SequentialSampler
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import set_seed, EarlyStopping, train, test
import torch.optim as optim
from models.spacial import BCINet
import json


def main():
    with open("params.json") as fp:
        params = json.load(fp)

    # Check cuda availability
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    set_seed(1234)

    dataset = concat_datasets(params["in_dir"])

    total_size = len(dataset)
    train_size = int(params["split_size"][0] * total_size)
    valid_size = int(params["split_size"][1] * total_size)
    test_size = total_size - train_size - valid_size

    indices = list(range(total_size))
    np.random.shuffle(indices)
    train_idx = indices[valid_size + test_size:]
    valid_idx = indices[:valid_size]
    test_idx = indices[valid_size:valid_size + test_size]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = SequentialSampler(valid_idx)
    test_sampler = SequentialSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], num_workers = 2, sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], num_workers=2, sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], num_workers=2, sampler=test_sampler)

    # Train and test model
    model = BCINet().to(params['device'])
    early_stopping = EarlyStopping(patience=5)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    train(model, {'Train': trainloader, 'Valid': validloader}, optimizer, criterion, params, callback=early_stopping)
    test(model, testloader, criterion, params)

if __name__ == "__main__":
    main()
    