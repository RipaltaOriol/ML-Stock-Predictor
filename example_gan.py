# GAN is a bit overkill for the problem solved here.
# But all elements of a GAN implementation are present
# and it is small enough to train during the lecture.
import torch
from torch import nn

import numpy as np

import math
import matplotlib.pyplot as plt


### Data Generation ###

def make_smiley(n: int) -> torch.Tensor:
    """Generate an Nx2 'smiley face' point cloud: two Gaussian eyes + smile arc."""
    n_eye = round(n // 2 * 0.6)
    n_mouth = n - 2 * n_eye

    eye_sigma = torch.tensor([0.1, 0.28])  # horizontal, vertical
    left_eye = torch.randn(n_eye, 2) * eye_sigma + torch.tensor([-0.6, 0.])
    right_eye = torch.randn(n_eye, 2) * eye_sigma + torch.tensor([0.6, 0.])

    # Smile: arc from ~30deg to ~150deg
    theta = torch.rand(n_mouth) * (5 * math.pi / 6 - math.pi / 6) + math.pi / 6
    r = 1.5
    mouth = torch.stack(
        (r * torch.cos(theta), -0.5 - 0.8 * r * torch.sin(theta)),
        dim=1,
    ) + 0.05 * torch.randn(n_mouth, 2)

    pts = torch.cat([left_eye, right_eye, mouth], dim=0)
    perm = torch.randperm(n)
    return pts[perm]


train_data_length = 1024 * 2
train_data = make_smiley(train_data_length)
train_labels = torch.ones(train_data_length)
trainset = [(train_data[i], train_labels[i]) for i in range(train_data_length)]

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)


### Discriminator ###

# To avoid sparsity, which can lead to cycles, we use leaky ReLU.

discriminator = nn.Sequential(
    nn.Linear(2, 256),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.LeakyReLU(0.2),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
)


### Generator ###

# Latent space dimension needs to be high enough to capture all modes and avoid mode collapse
latent_space_dim = 8
def gen_latent_space_data(batch_size):
    return torch.randn((batch_size, latent_space_dim))

generator = nn.Sequential(
    nn.Linear(latent_space_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)




### Optimizer ###

lr = 0.0002
loss_function = nn.BCEWithLogitsLoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))


### Training ###

def train_epoch():
    epoch_d_loss, epoch_g_loss = 0.0, 0.0
    discriminator.train()
    generator.train()

    num_batches = 0
    for n, (real_samples, real_samples_labels) in enumerate(train_loader):
        real_samples_labels = torch.reshape(real_samples_labels, (-1, 1))

        # Generate discriminator data
        current_bs = real_samples.size(0)
        latent_space_samples = gen_latent_space_data(current_bs)
        with torch.no_grad():
            generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((current_bs, 1))
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

        # Shuffle the combined samples and labels
        perm = torch.randperm(all_samples.size(0))
        all_samples = all_samples[perm]
        all_samples_labels = all_samples_labels[perm]

        # Training the discriminator
        # Some noise can be added to labels to avoid getting stuck by overfitting
        # In this case we have sufficient dropout to get a good solution
        discriminator.requires_grad_(True)
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = gen_latent_space_data(current_bs)

        # Training the generator
        discriminator.requires_grad_(False)
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)

        loss_generator = loss_function(
            output_discriminator_generated, real_samples_labels
        )
        loss_generator.backward()
        optimizer_generator.step()

        epoch_d_loss += loss_discriminator.item()
        epoch_g_loss += loss_generator.item()
        num_batches += 1

    return (
        epoch_d_loss / num_batches,
        epoch_g_loss / num_batches,
    )


def train_epochs(epochs, print_freq=10):
    last_d_loss, last_g_loss = None, None
    for epoch in epochs:
        epoch_d_loss, epoch_g_loss = train_epoch()
        last_d_loss, last_g_loss = epoch_d_loss, epoch_g_loss
        if print_freq != 0 and epoch % print_freq == 0:
            print(
                f"Epoch: {epoch+print_freq}\tLoss D.: {epoch_d_loss}\tLoss G.: {epoch_g_loss}"
            )
    return last_d_loss, last_g_loss


def plot_generator():
    latent_space_samples = gen_latent_space_data(1000)
    with torch.no_grad():
        generated_samples = generator(latent_space_samples)
    plt.scatter(train_data[:, 0], train_data[:, 1], s=6, c="lightgray", label="real")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], s=10, c="tab:blue", label="generated")
    plt.axis("equal")
    plt.legend(loc="best")


def plot_empirical(path: str = "empirical.png"):
    plt.figure(dpi=300)
    plt.scatter(train_data[:, 0], train_data[:, 1], s=8, c="black")
    plt.axis("equal")
    plt.title("Empirical data")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_and_visualize(num_epochs=500, plot_freq=10, save_images=False, display_inline=False):
    """Train the GAN with optional visualization.
    
    Args:
        num_epochs: Total number of epochs to train
        plot_freq: How often to plot/save progress
        save_images: Whether to save PNG files to disk
        display_inline: Whether to display plots inline (for notebooks)
    """
    from IPython import display as ipydisplay
    
    epochs = range(0, num_epochs + plot_freq, plot_freq)
    
    if save_images:
        plot_empirical("empirical.png")
    
    for start, stop in zip(epochs, epochs[1:]):
        if display_inline:
            ipydisplay.clear_output(wait=True)
        
        plt.figure(dpi=150 if display_inline else 300)
        d_loss, g_loss = train_epochs(range(start, stop), print_freq=0)
        plot_generator()
        plt.title(f"Epoch {stop}")
        
        if save_images:
            plt.savefig(f"epoch_{stop:03}.png")
        
        if display_inline:
            if d_loss is not None and g_loss is not None:
                print(f"Epoch {stop}/{num_epochs} - Loss D: {d_loss:.4f}, Loss G: {g_loss:.4f}")
            else:
                print(f"Training progress: Epoch {stop}/{num_epochs}")
            plt.show()
        else:
            plt.close()
    
    print("Finished training")


def main():
    """Main function for script execution."""
    num_epochs = 500
    plot_freq = 10
    epochs = range(0, num_epochs + plot_freq, plot_freq)

    plot_empirical("empirical.png")

    plot_generator()
    for start, stop in zip(epochs, epochs[1:]):
        plt.figure(dpi=300)
        train_epochs(range(start, stop))
        plot_generator()
        plt.savefig(f"epoch_{stop:03}.png")
        plt.close()
    print("Finished training")


if __name__ == "__main__":
    main()
