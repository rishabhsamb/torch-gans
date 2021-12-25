import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor

from PIL import Image


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.image_tensor[idx]

        image = Image.fromarray(image.detach().numpy(), mode='L')
        label = self.labels_tensor[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(image)

        return image, label


def create_dataloaders_mnist():

    train_dataset = torch.load("data/MNIST/processed/training.pt")
    val_dataset = torch.load("data/MNIST/processed/test.pt")

    # training_dataset = torch.utils.data.DataLoader(
    #     MNISTDataset(dataset=train_dataset, transform=ToTensor()), batch_size=32, shuffle=True)
    # test_set = torch.utils.data.DataLoader(
    #     MNISTDataset(dataset=val_dataset, transform=ToTensor()), batch_size=32, shuffle=True)

    # replacement=False: see http://shorturl.at/ejqtG
    train_dataset = MNISTDataset(dataset=train_dataset, transform=ToTensor())
    val_dataset = MNISTDataset(dataset=val_dataset, transform=ToTensor())

    training_dataset = torch.utils.data.DataLoader(torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples= ), )

    val_dataset = torch.utils.data.DataLoader(torch.utils.data.RandomSampler(val_dataset, replacement=False))

    return training_dataset, val_dataset
