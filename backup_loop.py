from functools import reduce
from dataloader import create_dataloaders_mnist
from generator import Generator
from discriminator import Discriminator
from torch.optim import Adam, SGD
import torch
from torch.nn import BCEWithLogitsLoss
import tqdm

from visualization import plot_train_graph, display_image_grid
from visualize_images import get_visualization

random_seed = 1729
torch.manual_seed(random_seed)

noise_prior = torch.distributions.normal.Normal(
    torch.tensor([0.0]), torch.tensor([1.0]))
minibatch_size = 32  # 'm' in Algorithm 1
noise_length = 100
epochs = 20
lr = 0.0005
criterion = BCEWithLogitsLoss()


def train(discriminator, generator, train, device):
    generator_history = []
    discriminator_history = []

    optimizer_generator = Adam(params=generator.parameters(), lr=lr)
    optimizer_discriminator = Adam(params=discriminator.parameters(), lr=lr)

    for i in tqdm.tqdm((range(epochs))):
        print(f"EPOCH {i+1} of {epochs}")
        for x in train:

            # Discriminator step
            x = torch.flatten(x, start_dim=1)
            x = x.to(device=device)

            ones = torch.ones((minibatch_size, 1)).to(device=device)
            zeroes = torch.zeros((minibatch_size, 1)).to(device=device)

            muh_noise = torch.randn(torch.Size(
                [minibatch_size, noise_length])).to(device=device)

            discriminator.zero_grad()
            real_loss = criterion(discriminator(x), ones)
            fake_loss = criterion(
                discriminator(generator(muh_noise)), zeroes)
            d_loss = real_loss + fake_loss
            # print(f"Discriminator Loss: {d_loss}")

            discriminator_history.append(d_loss.item())
            d_loss.backward()
            optimizer_discriminator.step()

            # Generator step
            generator.zero_grad()
            muh_noise_two = torch.randn(torch.Size(
                [minibatch_size, noise_length])).to(device=device)

            zeroes = torch.ones((minibatch_size, 1)).to(device=device)
            g_loss = criterion(
                discriminator(generator(muh_noise_two)), ones)
            # print(f"Generator loss: {g_loss}")

            generator_history.append(g_loss.item())
            g_loss.backward()
            optimizer_generator.step()

        if (i + 1) % (epochs // 5) == 0:
            print("wee wee")
            torch.save(generator.state_dict(), f'./generator-{i+1}')

    plot_train_graph(generator_history, discriminator_history)
    images = get_visualization(
        10, generator, noise_length=noise_length, device=device)
    display_image_grid(images=images)


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    generator = Generator(noise_length=noise_length).to(device=device)
    discriminator = Discriminator().to(device=device)

    train_set, _ = create_dataloaders_mnist(minibatch_size)

    # train_set = [next(iter(train_set))[0, :, :] for i in range(10)]

    # x = [torch.squeeze(img).cpu().detach().numpy() for img in iter(train_set)]

    # display_image_grid(images=x)

    # exit()

    train(discriminator, generator, train_set, device)
