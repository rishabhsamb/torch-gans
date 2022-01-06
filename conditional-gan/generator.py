import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
    Expected input shape: (minibatch_size, noise_length)
    Output shape: (minibatch_size, 784)
    """

    def __init__(self, noise_length, embedding_size):
        super().__init__()

        self.label_embedding = nn.Embedding(10, embedding_size)

        self.fc1 = nn.Linear(noise_length + embedding_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z, y):
        y_embed = self.label_embedding(y)
        # z: (batch, 100). Y_embed: (batch, 10)
        cat_input = torch.cat([z, y_embed], axis=1)
        x = self.fc1(cat_input)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.fc2(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.fc3(x)
        x = F.leaky_relu(x, 0.2, inplace=True)
        x = self.fc4(x)
        x = torch.tanh(x)
        return x
