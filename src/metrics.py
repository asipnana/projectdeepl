"""Standardized evaluation metrics: MSE, PSNR, SSIM, FID, and inference latency."""

import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.regression import MeanSquaredError


def rescale_to_unit(x: torch.Tensor, from_neg_one: bool = False) -> torch.Tensor:
    """Convert image tensor to [0, 1] range."""
    if from_neg_one:
        x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)


def to_fid_uint8_rgb(x: torch.Tensor, from_neg_one: bool = False) -> torch.Tensor:
    """Convert grayscale/RGB tensor to uint8 [0, 255] with 3 channels for InceptionV3."""
    x = rescale_to_unit(x, from_neg_one=from_neg_one)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return (x * 255.0).round().to(torch.uint8)


@torch.no_grad()
def evaluate_reconstruction(
    reconstruct_fn: Callable[[torch.Tensor], torch.Tensor],
    test_loader: DataLoader,
    device: torch.device,
    from_neg_one: bool = False,
) -> dict[str, float]:
    """
    Evaluate Test MSE, Test PSNR (dB), and Test SSIM on the entire Test Set.
    reconstruct_fn: takes a batch of images on `device` and returns reconstructed images.
    """
    mse_metric = MeanSquaredError().to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for images, _ in test_loader:
        images = images.to(device)
        recons = reconstruct_fn(images)

        real_unit = rescale_to_unit(images, from_neg_one=from_neg_one)
        recon_unit = rescale_to_unit(recons, from_neg_one=from_neg_one)

        mse_metric.update(recon_unit, real_unit)
        psnr_metric.update(recon_unit, real_unit)
        ssim_metric.update(recon_unit, real_unit)

    return {
        "mse": float(mse_metric.compute().item()),
        "psnr": float(psnr_metric.compute().item()),
        "ssim": float(ssim_metric.compute().item()),
    }


@torch.no_grad()
def compute_fid(
    test_loader: DataLoader,
    generate_fn: Callable[[int], torch.Tensor],
    device: torch.device,
    num_samples: int = 2048,
    gen_batch_size: int = 128,
    from_neg_one_real: bool = False,
    from_neg_one_fake: bool = False,
) -> float:
    """
    Compute FID between real Test Set images and `num_samples` generated images.
    Keep `num_samples` identical across VAE, Vanilla GAN, DCGAN, and DDPM for fairness.
    """
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # 1. Feed real Test Set images
    real_count = 0
    for images, _ in test_loader:
        images = images.to(device)
        fid.update(to_fid_uint8_rgb(images, from_neg_one=from_neg_one_real), real=True)
        real_count += images.shape[0]
        if real_count >= num_samples:
            break

    # 2. Feed generated synthetic images
    fake_count = 0
    while fake_count < num_samples:
        cur_bs = min(gen_batch_size, num_samples - fake_count)
        fake_imgs = generate_fn(cur_bs).to(device)
        fid.update(to_fid_uint8_rgb(fake_imgs, from_neg_one=from_neg_one_fake), real=False)
        fake_count += cur_bs

    return float(fid.compute().item())


@torch.no_grad()
def measure_latency_ms(
    inference_fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int = 5,
    runs: int = 20,
) -> dict[str, float]:
    """
    Measure inference latency (in ms) for a single batch of 64 samples.
    For slow iterative models (DDPM), pass smaller warmup/runs (e.g., warmup=1, runs=3).
    """
    for _ in range(warmup):
        _ = inference_fn()

    timings = []
    if device.type == "cuda":
        torch.cuda.synchronize()
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = inference_fn()
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
    else:
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = inference_fn()
            timings.append((time.perf_counter() - t0) * 1000.0)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
    }


if __name__ == "__main__":
    # Quick smoke test to verify all metric dependencies work in Colab
    from src.data import load_mnist
    from src.utils import get_device, set_seed

    set_seed(42)
    dev = get_device()
    _, _, test_loader = load_mnist(batch_size=128)

    # Dummy reconstruction test (adding tiny noise)
    recon_res = evaluate_reconstruction(
        lambda x: (x + 0.05 * torch.randn_like(x)).clamp(0.0, 1.0),
        test_loader,
        dev,
    )
    print("Smoke Test Reconstruction:", recon_res)

    # Dummy FID test (random noise vs real MNIST)
    fid_val = compute_fid(
        test_loader,
        lambda bs: torch.rand(bs, 1, 28, 28, device=dev),
        dev,
        num_samples=256,
    )
    print(f"Smoke Test FID (random noise): {fid_val:.2f}")

    # Dummy latency test (batch 64)
    lat = measure_latency_ms(lambda: torch.randn(64, 1, 28, 28, device=dev), dev)
    print(f"Smoke Test Latency (batch 64): {lat['mean_ms']:.3f} ± {lat['std_ms']:.3f} ms")