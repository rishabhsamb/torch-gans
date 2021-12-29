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
        self.fc1 = nn.Linear(784, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = F.dropout(x, 0.3)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)        
        x = F.dropout(x, 0.3)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2, inplace=True)    
        x = F.dropout(x, 0.3)
        x = self.fc4(x)
        return x
