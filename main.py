import json
import torch
from utils import set_seed, EarlyStopping, train, test, init_data_loaders
import torch.optim as optim
from models.spacial import BCINet
from models.temporal import TemporalModel_LSTM
from preprocessing.dataloaders import concat_datasets
from utils import count_parameters
from torch.utils.data import DataLoader



def main():
    with open("params.json") as fp:
        params = json.load(fp)

    # Check cuda availability
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    set_seed(1234)

    data_loaders = init_data_loaders(params)

    # Train and test model
    model = BCINet(in_channels=1, out_features=4, kernel_size=3, dropout=0.1)
    print("Model Number Parameters = ", count_parameters(model))
    early_stopping = EarlyStopping(patience=10)
    optimizer = optim.Adam(model.parameters(), params['lr'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train(model, {'Train': data_loaders['train'], 'Valid': data_loaders['validation']}, optimizer, criterion, params, callback=early_stopping)

    print()
    for state in data_loaders.keys():
        if state.startswith('test'):
            print("Results for this parameter: ", state[5:])
            test(model, data_loaders[state], criterion, params)
            print()

if __name__ == "__main__":
    main()
