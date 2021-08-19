import copy
import random

import matplotlib.pyplot as plt
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
            data_loaders['test_' + state] = DataLoader(datasets[state], batch_size=params["batch_size"], num_workers=2)

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


def plot_confusion(matrix, labels):
     num_labels = len(labels)
     norm_matrix = np.around(matrix / matrix.sum(axis=1)[:, np.newaxis], 2)

     im = plt.imshow(norm_matrix, cmap='Reds', vmin=0, vmax=1)
     ticks = range(num_labels)
     plt.xlabel('Predicted', fontsize=12)
     plt.ylabel('True Labels', fontsize=12)
     plt.xticks(ticks, labels)
     plt.yticks(ticks, labels)
     plt.title('Confusion Matrix', fontsize=20)
     for row in range(num_labels):
         for col in range(num_labels):
             color = 'white' if norm_matrix[row, col] > 0.5 else 'black'
             plt.text(col, row, norm_matrix[row, col], horizontalalignment='center', fontsize=12, color=color)
     plt.colorbar(im, fraction=0.046, pad=0.04)


def train(model, dataloaders, optimizer, criterion, params, callback=None):
    """Train the model for num_epochs using the trainloader and validloader."""
    example_input = torch.tensor(dataloaders['Train'].dataset.data[1:][0])[None, None, :, :]  # Unsqueeze input to (1, 1, 62, 500)
    example_input = example_input.to(params['device'])
    writer.add_graph(model, input_to_model=example_input)
    
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
                        loss.mean().backward()
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

    num_labels = 4
    confusion_matrix = np.zeros((num_labels, num_labels))  # Rows: Target, Columns: Predicted

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

            for i_label, label in enumerate(labels):
                 confusion_matrix[label, predicted_task[i_label]] += 1

     fig = plt.figure(figsize=(7, 7))
     plot_confusion(confusion_matrix, ['Right', 'Left', 'Up', 'Down'])

     writer.add_figure('Confusion Matrix', fig, 0)
     writer.close()

