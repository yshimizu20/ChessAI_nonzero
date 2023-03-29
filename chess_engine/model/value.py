import torch
import torch.nn as nn

from chess_engine.model.shared_layers import SharedConv


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.shared_conv = SharedConv()
        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.shared_conv(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x
