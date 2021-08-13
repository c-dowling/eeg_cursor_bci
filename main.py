import json

import torch
import torch.optim as optim

from models.spacial import BCINet
from utils import set_seed, EarlyStopping, train, test, init_data_loaders


def main():
    with open("params.json") as fp:
        params = json.load(fp)

    # Check cuda availability
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    set_seed(1234)

    trainloader, validloader, testloader = init_data_loaders(params)

    # Train and test model
    model = BCINet().to(params['device'])
    early_stopping = EarlyStopping(patience=5)
    optimizer = optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    train(model, {'Train': trainloader, 'Valid': validloader}, optimizer, criterion, params, callback=early_stopping)
    test(model, testloader, criterion, params)


if __name__ == "__main__":
    main()
