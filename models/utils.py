import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def set_seed(seed):
    """Sets rng using seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, dataloaders, optimizer, criterion, device, num_epochs=10, callback=None):
    """Train the model for num_epochs using the trainloader and validloader."""
    for epoch in range(num_epochs):
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
                inputs = inputs.to(device)
                labels = labels.to(device)

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

        if callback:
            validation_loss = epoch_loss / epoch_inputs
            if callback(model, validation_loss, epoch):
                break


def test(model, dataloaders, criterion, device):
    """Test the model using the testloader."""
    model.eval()
    test_loss = 0
    test_correct = 0
    test_inputs = 0

    bar = tqdm(enumerate(dataloaders['Test']), total=len(dataloaders['Test']), desc='Test: ')

    for batch, (inputs, labels) in bar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        probabilities_task = F.softmax(outputs, dim=1)
        _, predicted_task = torch.max(probabilities_task, 1)
        test_loss += loss.sum().item()
        test_correct += (predicted_task == labels).sum().item()
        test_inputs += len(labels)
        bar.set_postfix_str(f'Loss Test: {test_loss / test_inputs:.4f}, '
                            f'Acc Test: {test_correct / test_inputs:.4f}')
