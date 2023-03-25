import torch
import torch.nn as nn

from chess_engine.model.shared_layers import SharedConv


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.shared_conv = SharedConv()
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x
