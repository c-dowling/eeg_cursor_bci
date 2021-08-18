import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from preprocessing.dataloaders import concat_datasets, SequentialSampler
import copy


# Create writer to track training and testing evolution
writer = SummaryWriter()


def set_seed(seed):
    """Sets rng using seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def init_data_loaders(params):
    """
    Initialise all train, validation and test dataloaders
    args:
    params: parameters from params.json file
    """

    datasets = concat_datasets(params["in_dir"], params['is_temporal'])
    data_loaders = {}

    for state in datasets.keys():
        # Generate train and validation dataloaders
        if state == 'train':
            total_size = len(datasets[state])
            train_size = int(params["split_size"][0] * total_size)

            indices = list(range(total_size))
            np.random.shuffle(indices)

            train_idx = indices[:train_size]
            valid_idx = indices[train_size:]

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
            valid_sampler = SequentialSampler(valid_idx)

            data_loaders['train'] = DataLoader(datasets[state], batch_size=params["batch_size"], num_workers=2, sampler=train_sampler)
            data_loaders['validation'] = DataLoader(datasets[state], batch_size=params["batch_size"], num_workers=2, sampler=valid_sampler)

        # Create a test dataloader
        else:
            test_idx = len(datasets[state])
            test_sampler = SequentialSampler(test_idx)

            data_loaders['test_' + state] = DataLoader(datasets[state], batch_size=params["batch_size"], num_workers=2, sampler=test_sampler)

    return data_loaders


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.current_patience = patience
        self.best_model = None
        self.best_epoch = None
        self.best_loss = float('inf')

    def __call__(self, model, validation_loss, epoch):
        if validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model)
            self.current_patience = self.patience
        else:
            self.current_patience -= 1

        if self.current_patience == 0:
            print(f'Training halted. Retrieving weights from epoch {self.best_epoch}. Best validation loss {self.best_loss:.4f}')
            model.load_state_dict(self.best_model.state_dict())
            return True
        return False


def train(model, dataloaders, optimizer, criterion, params, callback=None):
    """Train the model for num_epochs using the trainloader and validloader."""
    for epoch in range(params['epochs']):
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_correct = 0
            epoch_inputs = 0
            bar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), desc=f'Epoch {epoch:>2} ({phase})')

            for batch, (inputs, labels) in bar:
                inputs = inputs.to(params['device'])
                labels = labels.to(params['device'])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
                
                probabilities_task = F.softmax(outputs, dim=1)
                _, predicted_task = torch.max(probabilities_task, 1)
                epoch_loss += loss.sum().item()
                epoch_correct += (predicted_task == labels).sum().item()
                
                epoch_inputs += len(labels)
                
                bar.set_postfix_str(f'Loss {phase}: {epoch_loss / epoch_inputs:.4f}, '
                                    f'Acc {phase}: {epoch_correct / epoch_inputs:.4f}')

            writer.add_scalar(f'Loss/{phase}', epoch_loss / epoch_inputs, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_correct / epoch_inputs, epoch)

        if callback:
            validation_loss = epoch_loss / epoch_inputs
            if callback(model, validation_loss, epoch):
                break


def test(model, dataloader, criterion, params):
    """Test the model using the testloader."""
    model.eval()
    test_loss = 0
    test_correct = 0
    test_inputs = 0

    bar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Test: ')
    with torch.no_grad():
        for _, (inputs, labels) in bar:
            inputs = inputs.to(params['device'])
            labels = labels.to(params['device'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            probabilities_task = F.softmax(outputs, dim=1)
            _, predicted_task = torch.max(probabilities_task, 1)
            test_loss += loss.sum().item()
            test_correct += (predicted_task == labels).sum().item()
            test_inputs += len(labels)
            bar.set_postfix_str(f'Loss Test: {test_loss / test_inputs:.4f}, '
                                f'Acc Test: {test_correct / test_inputs:.4f}')
