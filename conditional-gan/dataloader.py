import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor, Compose, Normalize
import torchvision.datasets as datasets
from PIL import Image


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset_images, _ = dataset

    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, idx):

        image = self.dataset_images[idx]

        image = Image.fromarray(image.detach().numpy(), mode='L')

        if self.transform:
            image = self.transform(image)

        return image


def create_dataloaders_mnist(batch_size):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    train_dataset = datasets.MNIST(
        root='./data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
