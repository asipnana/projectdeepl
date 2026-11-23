"""Train and evaluate DDPM (fast-track T=200, 15 epochs) on standardized MNIST."""

import argparse
import json
import os
import time

import torch

from src.data import load_mnist
from src.metrics import compute_fid, measure_latency_ms
from src.models.ddpm import DDPM
from src.utils import get_device, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--timesteps", type=int, default=200)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    train_loader, _, test_loader = load_mnist(batch_size=128, seed=42, normalize_neg_one=True)

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    res_dir = os.path.join(args.save_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    ddpm = DDPM(timesteps=args.timesteps).to(device)
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    loss_hist, epoch_times = [], []
    print(f"\n=== Training DDPM (T={args.timesteps}, Epochs={args.epochs}) ===")
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        ddpm.train()
        running = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = ddpm.training_loss(x)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        ep_time = time.perf_counter() - t0
        ep_loss = running / len(train_loader.dataset)
        loss_hist.append(ep_loss)
        epoch_times.append(ep_time)
        if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs:
            print(f"[DDPM] Epoch {epoch:02d}/{args.epochs} | Noise MSE Loss: {ep_loss:.5f} | Time: {ep_time:.2f}s")

    torch.save(ddpm.state_dict(), os.path.join(ckpt_dir, "ddpm.pt"))

    ddpm.eval()
    print("[DDPM] Mengukur inference latency (batch 64, iterative T steps)...")
    lat_m = measure_latency_ms(lambda: ddpm.sample(64, device), device, warmup=1, runs=3)

    print("[DDPM] Menghitung FID (2048 sampel)... mohon tunggu ~45-60 detik...")
    fid_val = compute_fid(
        test_loader,
        lambda bs: ddpm.sample(bs, device),
        device,
        num_samples=2048,
        gen_batch_size=256,
        from_neg_one_real=True,
        from_neg_one_fake=True,
    )
    print(f"==> [DDPM] FID: {fid_val:.2f} | Latency(64): {lat_m['mean_ms']:.2f} ms")

    metrics = {
        "ddpm": {
            "latent_dim": "28x28",
            "timesteps": args.timesteps,
            "train_time_per_epoch_s": sum(epoch_times) / len(epoch_times),
            "latency_ms_batch64": lat_m["mean_ms"],
            "test_mse": None,
            "test_psnr": None,
            "test_ssim": None,
            "fid": fid_val,
            "loss_history": loss_hist,
        }
    }
    out_path = os.path.join(res_dir, "ddpm_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved DDPM checkpoint and metrics to: {out_path}")


if __name__ == "__main__":
    main()