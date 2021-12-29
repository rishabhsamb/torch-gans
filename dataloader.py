import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor, Compose, Normalize

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

    train_dataset = torch.load("data/training.pt")
    val_dataset = torch.load("data/test.pt")

    train_dataset = MNISTDataset(
        dataset=train_dataset, transform=ToTensor())

    val_dataset = MNISTDataset(
        dataset=val_dataset, transform=ToTensor())

    training_dataset=torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True)

    val_dataset=torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = True)
    return training_dataset, val_dataset


if __name__ == "__main__":
    train, _= create_dataloaders_mnist(batch_size = 32)

    storage=[]
    for x in train:
        storage.append(torch.flatten(x))

    storage=torch.stack(storage)

    print(storage.mean())
    print(storage.std())
