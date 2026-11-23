"""Deep Convolutional GAN (DCGAN) tailored for 28x28 MNIST."""

import torch
import torch.nn as nn


def weights_init_dcgan(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class DCGANGenerator(nn.Module):
    """Pure convolutional generator: (B, 100) -> (B, 1, 28, 28)."""

    def __init__(self, latent_dim: int = 100):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # (B, 100, 1, 1) -> (B, 256, 7, 7)
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=7, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # (B, 256, 7, 7) -> (B, 128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # (B, 128, 14, 14) -> (B, 1, 28, 28) (no BatchNorm on output layer)
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z)


class DCGANDiscriminator(nn.Module):
    """Pure strided-convolution discriminator: (B, 1, 28, 28) -> (B, 1) logits."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, 28, 28) -> (B, 64, 14, 14) (no BatchNorm on input layer)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 64, 14, 14) -> (B, 128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # (B, 128, 7, 7) -> (B, 1, 1, 1)
            nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)