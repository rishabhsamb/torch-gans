import torch
from generator import Generator
import matplotlib.pyplot as plt
import os
def display_image_grid(images, val):
    fig, axs = plt.subplots(1, 10, figsize=(12, 12))
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(
        num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    for i in range(num_row * num_col):
        ax = axes[i//num_col, i % num_col]
        ax.imshow(images[i], cmap='Greys')
    plt.tight_layout()
    plt.savefig("images/{}.png".format(val))
    plt.close()

device = (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))

generator = Generator(100)
generator.load_state_dict(torch.load("generator.pt"))
generator = generator.to(device=device)

noise_prior = torch.distributions.uniform.Uniform(0, 1)
num_imgs = 10
noise_list = [torch.randn(torch.Size([1, 100])).to(device=device) for i in range(num_imgs)] 


for step in range(50):
    # generate
    print(step)

    images = [torch.squeeze(generator(noise).view(-1, 28, 28)).cpu().detach().numpy() for noise in noise_list]

    # Save
    display_image_grid(images, step)

    # Increment
    noise_list_change = [torch.randn(torch.Size([1, 100])).to(device=device) for i in range(num_imgs)] 

    noise_list = [noise_list[i] + 0.5*noise_list_change[i] for i in range(num_imgs)]

