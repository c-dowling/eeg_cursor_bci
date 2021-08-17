import torch
from utils import set_seed, EarlyStopping, train, test, init_data_loaders
import torch.optim as optim
from models.spacial import BCINet
from models.temporal import TemporalModel_LSTM
import json


def main():
    with open("params.json") as fp:
        params = json.load(fp)

    # Check cuda availability
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    set_seed(1234)

    trainloader, validloader, testloader = init_data_loaders(params)

    # Train and test model
    model = TemporalModel_LSTM(
        channels = 62,
        window = 500,
        hidden_size = 20,
        C = 4,
        num_layers = 1).to(params['device'])
    early_stopping = EarlyStopping(patience=25)
    optimizer = optim.Adam(model.parameters(), params['lr'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    train(model, {'Train': trainloader, 'Valid': validloader}, optimizer, criterion, params, callback=early_stopping)
    test(model, testloader, criterion, params)

if __name__ == "__main__":
    main()
    