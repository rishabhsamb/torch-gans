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
visualization_samples = 10

# noise_prior = torch.distributions.normal.Normal(
#     torch.tensor([0.0]), torch.tensor([1.0]))
discriminator_steps = 1  # 'k' in Algorithm 1
minibatch_size = 100  # 'm' in Algorithm 1
noise_length = 100
embedding_size = 10
epochs = 200
lr = 0.0002
criterion = BCEWithLogitsLoss()


def train_discriminator(minibatch_size, noise_length, x, y, generator, discriminator, discriminator_history, optimizer_discriminator, device):
    ones = torch.ones((minibatch_size, 1)).to(device=device)
    zeroes = torch.zeros((minibatch_size, 1)).to(device=device)

    muh_noise = torch.randn(torch.Size(
        [minibatch_size, noise_length])).to(device=device)
    discriminator.zero_grad()

    real_loss = criterion(discriminator(x, y), ones)

    fake_labels = torch.randint(0, 10, torch.Size(
        [minibatch_size, ]), dtype=torch.long).to(device=device)
    fake_loss = criterion(
        discriminator(generator(muh_noise, fake_labels), fake_labels), zeroes)

    d_loss = real_loss + fake_loss

    discriminator_history.append(d_loss.item())
    d_loss.backward()
    optimizer_discriminator.step()


def train_generator(minibatch_size, noise_length, generator, discriminator, generator_history, optimizer_generator, device):
    generator.zero_grad()
    muh_noise_two = torch.randn(torch.Size(
        [minibatch_size, noise_length])).to(device=device)
    fake_labels = torch.randint(0, 10, torch.Size(
        [minibatch_size, ]), dtype=torch.long).to(device=device)

    ones = torch.ones((minibatch_size, 1)).to(device=device)
    g_loss = criterion(
        discriminator(generator(muh_noise_two, fake_labels), fake_labels), ones)

    generator_history.append(g_loss.item())
    g_loss.backward()
    optimizer_generator.step()


def train(discriminator, generator, train, device):
    generator_history = []
    discriminator_history = []

    optimizer_generator = Adam(params=generator.parameters(), lr=lr)
    optimizer_discriminator = Adam(params=discriminator.parameters(), lr=lr)

    for i in tqdm.tqdm(range(epochs)):
        for x, y in tqdm.tqdm(train):
            x = torch.flatten(x, start_dim=1)
            x = x.to(device=device)
            y = y.to(device=device)

            train_discriminator(minibatch_size, noise_length, x, y, generator,
                                discriminator, discriminator_history, optimizer_discriminator, device)

            train_generator(minibatch_size, noise_length, generator,
                            discriminator, generator_history, optimizer_generator, device)
    plot_train_graph(generator_history, discriminator_history)
    torch.save(generator.state_dict(), './generator.pt')
    images = get_visualization(
        visualization_samples, generator, noise_length=noise_length, device=device)
    display_image_grid(images=[img for img in images])


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    generator = Generator(noise_length=noise_length,
                          embedding_size=embedding_size).to(device=device)
    discriminator = Discriminator(
        embedding_size=embedding_size).to(device=device)

    train_set = create_dataloaders_mnist(minibatch_size)

    train(discriminator, generator, train_set, device)
