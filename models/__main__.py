import torch
import torch.optim as optim
import torchvision

from .utils import set_seed, train, test
from .transforms import RandomWindow
from .dataset import load_bci_dataset
from .model import BCINet


# Set seed for reproducibility
set_seed(1234)

# Check cuda availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load dataset
transforms = torchvision.transforms.Compose([RandomWindow(width=500)])
dataloaders = load_bci_dataset(root=..., split_size=(0.8, 0.1, 0.1),
                               batch_size=32, transforms=transforms)

# Train and test model
model = BCINet()
optimizer = optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss(reduction='none')
train(model, dataloaders, optimizer, criterion, device, num_epochs=10)
test(model, dataloaders, device)
