"""Train and evaluate Vanilla AE, Sparse AE, and VAE on standardized MNIST."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data import load_mnist
from src.metrics import compute_fid, evaluate_reconstruction, measure_latency_ms
from src.models.sparse_ae import SparseAE
from src.models.vae import VAE, vae_loss_fn
from src.models.vanilla_ae import VanillaAE
from src.utils import get_device, set_seed


def train_and_eval_ae_family(save_dir: str, epochs: int = 25, lr: float = 1e-3, l1_lambda: float = 1e-3):
    set_seed(42)
    device = get_device()
    train_loader, _, test_loader = load_mnist(batch_size=128, seed=42, normalize_neg_one=False)

    ckpt_dir = os.path.join(save_dir, "checkpoints")
    res_dir = os.path.join(save_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Fixed batch of 64 images for AE latency measurement
    sample_x64, _ = next(iter(test_loader))
    sample_x64 = sample_x64[:64].to(device)

    all_metrics = {}
    mse_criterion = nn.MSELoss()

    # ---------------------------------------------------------
    # 1. Vanilla Autoencoder
    # ---------------------------------------------------------
    set_seed(42)
    vanilla_ae = VanillaAE(latent_dim=64).to(device)
    opt = torch.optim.Adam(vanilla_ae.parameters(), lr=lr)
    loss_hist, epoch_times = [], []

    print("\n=== Training Vanilla Autoencoder ===")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        vanilla_ae.train()
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon, _ = vanilla_ae(x)
            loss = mse_criterion(recon, x)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        ep_time = time.perf_counter() - t0
        ep_loss = running / len(train_loader.dataset)
        loss_hist.append(ep_loss)
        epoch_times.append(ep_time)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[Vanilla AE] Epoch {epoch:02d}/{epochs} | Loss: {ep_loss:.5f} | Time: {ep_time:.2f}s")

    torch.save(vanilla_ae.state_dict(), os.path.join(ckpt_dir, "vanilla_ae.pt"))
    vanilla_ae.eval()
    recon_m = evaluate_reconstruction(lambda x: vanilla_ae(x)[0], test_loader, device)
    lat_m = measure_latency_ms(lambda: vanilla_ae(sample_x64)[0], device)
    all_metrics["vanilla_ae"] = {
        "latent_dim": 64,
        "train_time_per_epoch_s": sum(epoch_times) / len(epoch_times),
        "latency_ms_batch64": lat_m["mean_ms"],
        "test_mse": recon_m["mse"],
        "test_psnr": recon_m["psnr"],
        "test_ssim": recon_m["ssim"],
        "fid": None,
        "loss_history": loss_hist,
    }
    print(f"==> [Vanilla AE] Test: {recon_m} | Latency(64): {lat_m['mean_ms']:.3f} ms")

    # ---------------------------------------------------------
    # 2. Sparse Autoencoder
    # ---------------------------------------------------------
    set_seed(42)
    sparse_ae = SparseAE(latent_dim=64).to(device)
    opt = torch.optim.Adam(sparse_ae.parameters(), lr=lr)
    loss_hist, epoch_times = [], []

    print("\n=== Training Sparse Autoencoder ===")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        sparse_ae.train()
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon, h = sparse_ae(x)
            mse_loss = mse_criterion(recon, x)
            l1_penalty = h.abs().mean()
            loss = mse_loss + l1_lambda * l1_penalty
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        ep_time = time.perf_counter() - t0
        ep_loss = running / len(train_loader.dataset)
        loss_hist.append(ep_loss)
        epoch_times.append(ep_time)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[Sparse AE]  Epoch {epoch:02d}/{epochs} | Total Loss: {ep_loss:.5f} | Time: {ep_time:.2f}s")

    torch.save(sparse_ae.state_dict(), os.path.join(ckpt_dir, "sparse_ae.pt"))
    sparse_ae.eval()
    recon_m = evaluate_reconstruction(lambda x: sparse_ae(x)[0], test_loader, device)
    lat_m = measure_latency_ms(lambda: sparse_ae(sample_x64)[0], device)

    # Measure actual latent sparsity (% of exact zeros on Test Set)
    zero_count, total_elems = 0, 0
    with torch.no_grad():
        for x, _ in test_loader:
            h = sparse_ae.encode(x.to(device))
            zero_count += (h == 0).sum().item()
            total_elems += h.numel()
    sparsity_pct = 100.0 * zero_count / total_elems

    all_metrics["sparse_ae"] = {
        "latent_dim": 64,
        "l1_lambda": l1_lambda,
        "latent_sparsity_pct": sparsity_pct,
        "train_time_per_epoch_s": sum(epoch_times) / len(epoch_times),
        "latency_ms_batch64": lat_m["mean_ms"],
        "test_mse": recon_m["mse"],
        "test_psnr": recon_m["psnr"],
        "test_ssim": recon_m["ssim"],
        "fid": None,
        "loss_history": loss_hist,
    }
    print(f"==> [Sparse AE]  Test: {recon_m} | Sparsity: {sparsity_pct:.1f}% zeros | Latency(64): {lat_m['mean_ms']:.3f} ms")

    # ---------------------------------------------------------
    # 3. Variational Autoencoder (VAE)
    # ---------------------------------------------------------
    set_seed(42)
    vae = VAE(latent_dim=64).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    loss_hist, epoch_times = [], []

    print("\n=== Training Variational Autoencoder (VAE) ===")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        vae.train()
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            recon, mu, logvar = vae(x)
            loss = vae_loss_fn(recon, x, mu, logvar)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        ep_time = time.perf_counter() - t0
        ep_loss = running / len(train_loader.dataset)
        loss_hist.append(ep_loss)
        epoch_times.append(ep_time)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[VAE]        Epoch {epoch:02d}/{epochs} | ELBO Loss: {ep_loss:.2f} | Time: {ep_time:.2f}s")

    torch.save(vae.state_dict(), os.path.join(ckpt_dir, "vae.pt"))
    vae.eval()
    recon_m = evaluate_reconstruction(lambda x: vae(x)[0], test_loader, device)
    # Measure single-pass generative latency for 64 samples (for Q3 comparison vs GAN/DDPM)
    lat_m = measure_latency_ms(lambda: vae.sample(64, device), device)
    fid_val = compute_fid(
        test_loader,
        lambda bs: vae.sample(bs, device),
        device,
        num_samples=2048,
        from_neg_one_real=False,
        from_neg_one_fake=False,
    )
    all_metrics["vae"] = {
        "latent_dim": 64,
        "train_time_per_epoch_s": sum(epoch_times) / len(epoch_times),
        "latency_ms_batch64": lat_m["mean_ms"],
        "test_mse": recon_m["mse"],
        "test_psnr": recon_m["psnr"],
        "test_ssim": recon_m["ssim"],
        "fid": fid_val,
        "loss_history": loss_hist,
    }
    print(f"==> [VAE]        Test: {recon_m} | FID: {fid_val:.2f} | Latency(64): {lat_m['mean_ms']:.3f} ms")

    out_path = os.path.join(res_dir, "ae_family_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved AE family checkpoints and metrics to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()
    train_and_eval_ae_family(args.save_dir, epochs=args.epochs)