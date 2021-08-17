import json
import torch
from utils import set_seed, EarlyStopping, train, test, init_data_loaders
import torch.optim as optim
from models.spacial import BCINet
from models.temporal import TemporalModel_LSTM
import json
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

    trainloader, validloader, testloader = init_data_loaders(params, isTemporal=True)
    

    # Train and test model
    model = TemporalModel_LSTM(
        channels = 62,
        window = 40,
        hidden_size = 20,
        C = 4,
        num_layers = 1).to(params['device'])
    print("Model Number Parameters = {}".format(count_parameters(model)))
    early_stopping = EarlyStopping(patience=10)
    optimizer = optim.Adam(model.parameters(), params['lr'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    train(model, {'Train': trainloader, 'Valid': validloader}, optimizer, criterion, params, callback=early_stopping)
    del trainloader, validloader

    test(model, testloader, criterion, params)
    del testloader

    dataset = concat_datasets(params["sess2_dir"], True)
    testloader_session2 = DataLoader(dataset, batch_size=params["batch_size"], num_workers=2)
    test(model, testloader_session2, criterion, params)


if __name__ == "__main__":
    main()
