"""MLP Classifier and Frozen Latent Decoder for empirical limitation proof."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    """784 -> 256 -> 64 -> 10 classifier mirroring an Autoencoder encoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, num_classes)

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x_flat))
        z_mlp = F.relu(self.fc2(h1))
        return z_mlp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mlp = self.extract_latent(x)
        logits = self.fc3(z_mlp)
        return logits

    @torch.no_grad()
    def transpose_from_class(self, y_onehot: torch.Tensor) -> torch.Tensor:
        """
        Tahap 2: Top-Down Weight Transposition from one-hot class vector y_c:
        z_hat = ReLU(W3^T y_c) -> h1_hat = ReLU(W2^T z_hat) -> x_hat = Sigmoid(W1^T h1_hat)
        Note: PyTorch nn.Linear weight has shape (out_features, in_features),
        so row-vector multiplication (v @ weight) implements W^T v.
        """
        z_hat = F.relu(y_onehot @ self.fc3.weight)
        h1_hat = F.relu(z_hat @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)

    @torch.no_grad()
    def transpose_from_latent(self, z_mlp: torch.Tensor) -> torch.Tensor:
        """
        Tahap 3: Latent Transpose Projection from extracted z_MLP:
        x_hat = Sigmoid(W1^T ReLU(W2^T z_MLP))
        """
        h1_hat = F.relu(z_mlp @ self.fc2.weight)
        x_hat = torch.sigmoid(h1_hat @ self.fc1.weight)
        return x_hat.view(-1, 1, 28, 28)


class FrozenLatentDecoder(nn.Module):
    """Tahap 4: Separate MLP Decoder (64 -> 256 -> 784) trained on frozen z_MLP."""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, output_dim: int = 784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(-1, 1, 28, 28)