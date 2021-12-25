from dataloader import create_dataloaders_mnist
from generator import Generator
from discriminator import Discriminator
import torch
random_seed = 1729
torch.manual_seed(random_seed)

noise_prior = torch.distributions.uniform.Uniform(0, 1)
discriminator_steps = 1  # 'k' in Algorithm 1
minibatch_size = 32  # 'm' in Algorithm 1
noise_length = 32
epochs = 20


def train(discriminator, generator, train, device):
    for i in range(epochs):
        for step in range(discriminator_steps):
            muh_noise = noise_prior.sample(sample_shape=torch.Size(
                [minibatch_size, noise_length])).to(device=device)

            muh_samples = train. .to(device=device)
            # update SGD with custom loss
        muh_noise_two = noise_prior.sample(sample_shape=torch.Size(
            [minibatch_size, noise_length])).to(device=device)


if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    print(f"Training on device {device}.")

    generator = Generator().to(device=device)
    discriminator = Discriminator().to(device=device)

    train_set, _ = create_dataloaders_mnist()

    train(discriminator, generator, train_set, device)
