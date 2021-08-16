import torch.nn as nn
import torch


class BCINet(nn.Module):
    """From Schirrmeister et al., 2017 (https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/hbm.23730)
       Alternatively, use: https://robintibor.github.io/braindecode/index.html

    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=40, kernel_size=(1, 26)),  # Temporal Conv.
            nn.BatchNorm2d(num_features=40),
            nn.ELU(),  # TODO: Add dropout
            nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(62, 1)),  # Spatial Conv.
            nn.BatchNorm2d(num_features=40),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 56), stride=(1, 14)),
            nn.Flatten(start_dim=1),
            nn.Linear(1200, 4)
        )

    def forward(self, x):
        return self.model(x)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout):
        """
        Temporal convolution block based on a residual block from Sulabh Kumra and Christopher Kanan 2017
        https://www.researchgate.net/publication/310953119_Robotic_Grasp_Detection_using_Deep_Convolutional_Neural_Networks
        Also see this: https://unit8.co/resources/temporal-convolutional-networks-and-forecasting/
        """
        super().__init__()
        # Padding on the left side
        pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        # Convolutional network (we didnt include weight norms)
        conv2d1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                            stride=stride, dilation=dilation)
        elu = nn.ELU()
        dropout = nn.Dropout(dropout)
        conv2d2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                            stride=stride, dilation=dilation)
        self.net = nn.Sequential(pad, conv2d1, elu, dropout, pad, conv2d2, elu, dropout)

    def forward(self, x):
        return self.net(x)


# Shallow ConvNet (don't mistake with residual ConvNet)
class ShallowNet(nn.Module):
    """
        From Schirrmeister et al., 2017 (https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/hbm.23730)
        Alternatively, use: https://robintibor.github.io/braindecode/index.html
    """
    def __init__(self, in_channels, out_features, kernel_size, dropout):
        super().__init__()
        tcn1 = TCNBlock(in_channels=in_channels, out_channels=40, kernel_size=kernel_size, stride=1,
                        padding=kernel_size-1, dilation=1, dropout=dropout)
        spatial_filter_1 = nn.Conv2d(in_channels=40, out_channels=40, kernel_size=(62, 1))
        batch_norm = nn.BatchNorm2d(num_features=40)
        elu = nn.ELU()
        dropout = nn.Dropout(dropout)
        avg2dpool1 = nn.AvgPool2d(kernel_size=(1, 56), stride=(1, 14))
        flatten = nn.Flatten(start_dim=1)
        linear1 = nn.Linear(in_features=1280, out_features=out_features)
        self.model = nn.Sequential(tcn1, spatial_filter_1, batch_norm, elu, dropout, avg2dpool1, flatten, linear1)

    def forward(self, x):
        return self.model(x)

