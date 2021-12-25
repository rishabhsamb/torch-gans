import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    """
    Expected input shape: (minibatch_size, 784)
    Output shape: (minibatch_size, 1)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x
