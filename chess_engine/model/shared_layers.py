import torch
import torch.nn as nn


class SharedConv(nn.Module):
    def __init__(self):
        super(SharedConv, self).__init__()
        self.conv1 = nn.Conv2d(18, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.flatten(x)
        return x
