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
from preprocessing.dataloaders import CustomSplitTemporalDataset



def main():
    with open("params.json") as fp:
        params = json.load(fp)

    # Check cuda availability
    params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set seed for reproducibility
    set_seed(1234)

    #trainloader, validloader, testloader = init_data_loaders(params, isTemporal=True)
    datasets = CustomSplitTemporalDataset(params["in_dir"])
    train_loader = DataLoader(datasets.train,
                            batch_size=params["batch_size"],
                            num_workers=2,
                            shuffle=True)
    test_lr_loader = DataLoader(datasets.test_lr,
                            batch_size=params["batch_size"],
                            num_workers=2)
    test_ud_loader = DataLoader(datasets.test_ud,
                            batch_size=params["batch_size"],
                            num_workers=2)
    test_2d_loader = DataLoader(datasets.test_2d,
                            batch_size=params["batch_size"],
                            num_workers=2)


    # Train and test model
    model = TemporalModel_LSTM(
        channels = 62,
        window = 40,
        hidden_size = 20,
        C = 4,
        num_layers = 1).to(params['device'])

    
    
    early_stopping = EarlyStopping(patience=5)
    optimizer = optim.Adam(model.parameters(), params['lr'])
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    
    print("TRAINING MODEL ...")
    train(model, {'Train':  train_loader}, optimizer, criterion, params, callback=early_stopping)

    print("TESTING MODEL ON LR TRIALS ...")
    test(model, test_lr_loader, criterion, params)

    print("TESTING MODEL ON UD TRIALS ...")
    test(model, test_ud_loader, criterion, params)

    print("TESTING MODEL ON 2D TRIALS ...")
    test(model, test_2d_loader, criterion, params)
    
    
    print("MODEL NUMBER PARAMETERS : {}".format(count_parameters(model)))


if __name__ == "__main__":
    main()
