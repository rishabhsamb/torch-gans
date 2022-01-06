import torch


def get_visualization(n, generator, noise_length, device):
    images = []

    for i in range(n):
        muh_noise = torch.randn(torch.Size(
            [1, noise_length])).to(device=device)
        muh_label = torch.tensor([i], dtype=torch.long).to(device=device)

        gen_img = torch.squeeze(
            generator(muh_noise, muh_label).view(-1, 28, 28))

        images.append(gen_img.cpu().detach().numpy())

    return images
