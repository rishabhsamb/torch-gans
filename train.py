from functools import reduce
from dataloader import create_dataloaders_mnist
from generator import Generator
from discriminator import Discriminator
from torch.optim import Adam, SGD
import torch
from torch.nn import BCELoss

from visualization import plot_train_graph, display_image_grid
from visualize_images import get_visualization

random_seed = 1729
torch.manual_seed(random_seed)

noise_prior = torch.distributions.normal.Normal(
    torch.tensor([0.0]), torch.tensor([1.0]))
discriminator_steps = 1  # 'k' in Algorithm 1
minibatch_size = 100  # 'm' in Algorithm 1
noise_length = 3
epochs = 200
lr = 0.0005
criterion = BCELoss()


def train(discriminator, generator, train, device):
    generator_history = []
    discriminator_history = []

    optimizer_generator = Adam(params=generator.parameters(), lr=lr)
    optimizer_discriminator = Adam(params=discriminator.parameters(), lr=lr)

    for i in (range(epochs)):
        print(f"EPOCH {i+1} of {epochs}")
        for _ in range(discriminator_steps):
            # sample
            x = next(iter(train))
            x = torch.flatten(x, start_dim=1)
            x = x.to(device=device)

            # real/fake labels
            ones = torch.ones((minibatch_size, 1)).to(device=device)
            zeroes = torch.zeros((minibatch_size, 1)).to(device=device)

            # noise vectors (first sample)
            muh_noise = torch.randn(torch.Size(
                [minibatch_size, noise_length])).to(device=device)

            # discriminator loss on real criteria
            real_loss = criterion(discriminator(x), ones)

            # discriminator loss on fake criteria
            fake_loss = criterion(
                discriminator(generator(muh_noise)), zeroes)

            # final discriminator loss
            d_loss = real_loss + fake_loss
            print(f"Discriminator Loss: {d_loss}")

        # refresh weights
        discriminator.zero_grad()

        # backpropagate
        d_loss.backward()
        optimizer_discriminator.step()

        # for loss plot
        discriminator_history.append(d_loss.item())

        # noise vectors (second sample)
        muh_noise_two = torch.randn(torch.Size(
            [minibatch_size, noise_length])).to(device=device)

        # compute generator loss
        g_loss = criterion(
            discriminator(generator(muh_noise_two)), ones)
        print(f"Generator loss: {g_loss}")

        # refresh weights
        generator.zero_grad()

        # backpropagate
        g_loss.backward()
        optimizer_generator.step()

        # for loss plot
        generator_history.append(g_loss.item())

    plot_train_graph(generator_history, discriminator_history)
    torch.save(generator.state_dict(), './generator')
    images = get_visualization(
        10, generator, noise_length=noise_length, device=device)
    display_image_grid(images=images)


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    generator = Generator(noise_length=noise_length).to(device=device)
    discriminator = Discriminator().to(device=device)

    train_set, _ = create_dataloaders_mnist(
        epochs*minibatch_size, minibatch_size)

    # train_set = [next(iter(train_set))[0, :, :] for i in range(10)]

    # x = [torch.squeeze(img).cpu().detach().numpy() for img in iter(train_set)]

    # display_image_grid(images=x)

    # exit()

    train(discriminator, generator, train_set, device)
