import torch
import torch.nn as nn

from chess_engine.model.shared_layers import SharedConv


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.shared_conv = SharedConv()
        self.fc1 = nn.Linear(8 * 8 * 128, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.relu3 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(2048, 1968)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        x = self.shared_conv(x)
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        if mask is not None:
            x = x * mask
        x = self.sm(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            x = x.view(1, 18, 8, 8)
            return self.forward(x)
