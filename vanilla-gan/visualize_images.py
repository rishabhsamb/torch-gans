import torch


def get_visualization(n, generator, noise_length, device):
    images = []
    noise_prior = torch.distributions.uniform.Uniform(0, 1)

    for _ in range(n):
        muh_noise = torch.randn(torch.Size(
            [1, noise_length])).to(device=device)

        gen_img = torch.squeeze(generator(muh_noise).view(-1, 28, 28))

        images.append(gen_img.cpu().detach().numpy())

    return images
