import matplotlib.pyplot as plt

import os


def plot_train_graph(generator_history, discriminator_history):
    plt.plot([i+1 for i in range(len(generator_history))],
             generator_history, label="generator loss")
    plt.plot([i+1 for i in range(len(discriminator_history))],
             discriminator_history, label="discriminator loss")
    plt.title("loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend()

    if os.path.exists("./test.png"):
        os.remove("./test.png")
    plt.savefig("test.png")


def display_image_grid(images):
    fig, axs = plt.subplots(1, 10, figsize=(12, 12))
    if os.path.exists("./images.png"):
        os.remove("./images.png")
    num_row = 2
    num_col = 5
    fig, axes = plt.subplots(
        num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    for i in range(num_row * num_col):
        ax = axes[i//num_col, i % num_col]
        ax.imshow(images[i], cmap='gray')
    plt.tight_layout()
    plt.savefig("images.png")
    # for i, img in enumerate(images):

    #     axs[i].imshow(images[i], cmap="gray")
    #     plt.savefig("image-{i}.png")
