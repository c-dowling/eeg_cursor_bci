from torch import nn
import torch.nn.functional as F
from torch import flatten
from torch.nn.modules.dropout import Dropout

class Block(nn.Module):
    #expansion = 1
    def __init__(self, in_channels, out_channels, kernel_size, i_downsample=None, padding=0, stride=1):
        super(Block, self).__init__()

        self.projected_shortcut = False
        self.dropout = nn.Dropout2d()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding = padding, stride=stride, bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding = padding, stride=stride, bias=False)

        self.i_downsample = i_downsample
        self.stride = stride

        if(in_channels != out_channels):
            self.projected_shortcut = True
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)


    def forward(self, x):
        identity = x.clone()
        x = self.dropout(x)
        x = F.relu(self.batch_norm(self.conv1(x)))
        x = self.dropout(x)
        x = self.batch_norm(self.conv2(x))

        if(self.projected_shortcut):
            identity = self.projection(identity)
        x += identity
        return x

class ChannelFeatureExtractor(nn.Module):
    def __init__(self):
        super(ChannelFeatureExtractor, self).__init__()
        #self.conv1 = nn.Conv2d(1, 8, kernel_size=(1,7), padding=(0,3))
        self.block1 = Block(1, 4, (1,3), padding=(0,1))
        #self.block2 = Block(16, 16, (1,3), padding=(0,1))
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(16)

    def forward(self, x):
        #x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.block1(x)))
        #x = F.relu(self.bn3(self.block2(x)))
        return x


class SpatialFeatureExtractor(nn.Module):
    def __init__(self):
        super(SpatialFeatureExtractor, self).__init__()
        self.block1 = Block(1, 4, 3, padding=1)
        #self.block2 = Block(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.block1(x)))
        #x = F.relu(self.bn2(self.block2(x)))
        return x

class Classifier(nn.Module):
    def __init__(self, in_features, C, isTwoHead = False):
        super(Classifier, self).__init__()
        self.isTwoHead = isTwoHead
        if isTwoHead:
            ## TODO: Assert  C to be a list
            self.h1 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200,C[0])
            )
            self.h2 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features, 200),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(200,C[1])
            )
        else:
            ## TODO: Assert  C to be a positive integer
            self.dropout = Dropout()
            #self.fc1 = nn.Linear(in_features, 200)
            self.fc2 = nn.Linear(in_features, C)

    def forward(self,x):
        if self.isTwoHead:
            o1 = self.h1(x)
            o2 = self.h2(x)
            return o1, o2
        else:
            #x = self.dropout(x)
            #x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x


class TemporalModel_LSTM(nn.Module):
    def __init__(self, channels, window, hidden_size, C, num_layers=1, twoHead = False):
        super(TemporalModel_LSTM, self).__init__()
        self.C = C
        self.channelFeatureExtractor = ChannelFeatureExtractor()
        self.avg_pool = nn.AvgPool2d((1,2), stride=(1,2))
        self.spatialFeatureExtractor = SpatialFeatureExtractor()
        self.temporalFeatureExtractor = nn.LSTM(channels*window//2, hidden_size, num_layers, bidirectional=True, batch_first = True, dropout=0.5)
        self.classifier = Classifier(hidden_size*num_layers*2, self.C, isTwoHead=twoHead)

    def forward(self, x):
        #x = self.channelFeatureExtractor(x)
        #x = self.avg_pool(x)
        x = self.spatialFeatureExtractor(x)
        x = self.avg_pool(x)
        x = flatten(x,2)
        _, hidden = self.temporalFeatureExtractor(x)
        del x
        hidden = hidden[0].permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        hidden = self.classifier(hidden)

        return hidden
