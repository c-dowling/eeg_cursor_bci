import torch.nn as nn


class BCINet(nn.Module):
    """From Schirrmeister et al., 2017 (https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/hbm.23730)"""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 26)),  # Temporal Conv.
            nn.BatchNorm2d(num_features=40),
            nn.ELU(),
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(62, 1)),  # Spatial Conv.
            nn.BatchNorm2d(num_features=40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 56), stride=(1, 14)),
            nn.Flatten(start_dim=1),
            nn.Linear(1200, 4)
        )

    def forward(self, x):
        return self.model(x)
