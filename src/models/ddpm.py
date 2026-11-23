"""Lightweight U-Net and DDPM noise scheduler for 28x28 MNIST."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LightweightUNet(nn.Module):
    """Lightweight U-Net (32 -> 64 -> 128 -> 64 -> 32 -> 1) predicting noise eps_theta(x_t, t)."""

    def __init__(self, time_dim: int = 64):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        # Downsampling path
        self.d1_conv = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.d1_gn = nn.GroupNorm(8, 32)
        self.d1_t = nn.Linear(time_dim, 32)

        self.d2_conv = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14x14
        self.d2_gn = nn.GroupNorm(8, 64)
        self.d2_t = nn.Linear(time_dim, 64)

        # Bottleneck (7x7)
        self.bot_conv1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bot_gn1 = nn.GroupNorm(8, 128)
        self.bot_t = nn.Linear(time_dim, 128)
        self.bot_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bot_gn2 = nn.GroupNorm(8, 128)

        # Upsampling path
        self.u1_up = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 14x14
        self.u1_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.u1_gn = nn.GroupNorm(8, 64)
        self.u1_t = nn.Linear(time_dim, 64)

        self.u2_up = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 28x28
        self.u2_conv = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.u2_gn = nn.GroupNorm(8, 32)
        self.u2_t = nn.Linear(time_dim, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)

        h1 = F.silu(self.d1_gn(self.d1_conv(x)) + self.d1_t(t_emb)[:, :, None, None])
        h2 = F.silu(self.d2_gn(self.d2_conv(h1)) + self.d2_t(t_emb)[:, :, None, None])

        b = F.silu(self.bot_gn1(self.bot_conv1(h2)) + self.bot_t(t_emb)[:, :, None, None])
        b = F.silu(self.bot_gn2(self.bot_conv2(b)))

        u1 = torch.cat([self.u1_up(b), h2], dim=1)
        u1 = F.silu(self.u1_gn(self.u1_conv(u1)) + self.u1_t(t_emb)[:, :, None, None])

        u2 = torch.cat([self.u2_up(u1), h1], dim=1)
        u2 = F.silu(self.u2_gn(self.u2_conv(u2)) + self.u2_t(t_emb)[:, :, None, None])

        return self.out_conv(u2)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model with linear beta schedule."""

    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.04):
        super().__init__()
        self.timesteps = timesteps
        self.unet = LightweightUNet()

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def training_loss(self, x0: torch.Tensor) -> torch.Tensor:
        bs = x0.size(0)
        t = torch.randint(0, self.timesteps, (bs,), device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)

        sqrt_ab = self.sqrt_alphas_cumprod[t].view(bs, 1, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alphas_cumprod[t].view(bs, 1, 1, 1)
        x_t = sqrt_ab * x0 + sqrt_omab * noise

        pred_noise = self.unet(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def sample(self, num_samples: int, device: torch.device, return_trajectory: bool = False):
        x = torch.randn(num_samples, 1, 28, 28, device=device)
        capture_steps = {self.timesteps - 1, 150, 100, 60, 30, 10, 0}
        trajectory = []

        for step in reversed(range(self.timesteps)):
            t_batch = torch.full((num_samples,), step, device=device, dtype=torch.long)
            pred_noise = self.unet(x, t_batch)

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alphas_cumprod[step]
            beta_t = self.betas[step]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise
            )
            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

            if return_trajectory and step in capture_steps:
                trajectory.append((step, x.clamp(-1.0, 1.0).cpu()))

        x = x.clamp(-1.0, 1.0)
        if return_trajectory:
            return x, trajectory
        return x