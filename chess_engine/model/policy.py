import torch
import torch.nn as nn

from chess_engine.model.shared_layers import SharedConv


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.shared_conv = SharedConv()
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 1968 * 2)
        self.bn4 = nn.BatchNorm1d(1968 * 2)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(1968 * 2, 1968)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        x = self.shared_conv(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.fc2(x)
        if mask:
            x = x * mask
        x = self.sm(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            x = x.view(1, 18, 8, 8)
            return self.forward(x)
