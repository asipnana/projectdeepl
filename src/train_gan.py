"""Train and evaluate Vanilla GAN and DCGAN on standardized MNIST ([-1, 1])."""

import argparse
import json
import os
import time

import torch
import torch.nn as nn

from src.data import load_mnist
from src.metrics import compute_fid, measure_latency_ms
from src.models.dcgan import DCGANDiscriminator, DCGANGenerator, weights_init_dcgan
from src.models.vanilla_gan import VanillaDiscriminator, VanillaGenerator
from src.utils import get_device, set_seed


def train_single_gan(
    name: str,
    gen: nn.Module,
    disc: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    betas: tuple[float, float],
    ckpt_dir: str,
) -> dict:
    criterion = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    g_losses, d_losses, epoch_times = [], [], []
    latent_dim = gen.latent_dim

    print(f"\n=== Training {name} ===")
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        gen.train()
        disc.train()
        run_g, run_d = 0.0, 0.0

        for real_imgs, _ in train_loader:
            bs = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            real_labels = torch.ones(bs, 1, device=device)
            fake_labels = torch.zeros(bs, 1, device=device)

            # 1. Update Discriminator: max log(D(x)) + log(1 - D(G(z)))
            opt_d.zero_grad()
            out_real = disc(real_imgs)
            loss_real = criterion(out_real, real_labels)

            z = torch.randn(bs, latent_dim, device=device)
            fake_imgs = gen(z)
            out_fake = disc(fake_imgs.detach())
            loss_fake = criterion(out_fake, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            opt_d.step()

            # 2. Update Generator: max log(D(G(z)))
            opt_g.zero_grad()
            out_fake_for_g = disc(fake_imgs)
            loss_g = criterion(out_fake_for_g, real_labels)
            loss_g.backward()
            opt_g.step()

            run_d += loss_d.item() * bs
            run_g += loss_g.item() * bs

        ep_time = time.perf_counter() - t0
        avg_d = run_d / len(train_loader.dataset)
        avg_g = run_g / len(train_loader.dataset)
        d_losses.append(avg_d)
        g_losses.append(avg_g)
        epoch_times.append(ep_time)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[{name}] Epoch {epoch:02d}/{epochs} | D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} | Time: {ep_time:.2f}s")

    # Save checkpoints before FID evaluation
    torch.save(gen.state_dict(), os.path.join(ckpt_dir, f"{name}_gen.pt"))
    torch.save(disc.state_dict(), os.path.join(ckpt_dir, f"{name}_disc.pt"))

    # Evaluate Latency (batch 64) & FID (2048 samples)
    gen.eval()
    lat_m = measure_latency_ms(lambda: gen.sample(64, device), device)
    print(f"[{name}] Menghitung FID (2048 sampel)... mohon tunggu ~30-60 detik...")
    fid_val = compute_fid(
        test_loader,
        lambda bs: gen.sample(bs, device),
        device,
        num_samples=2048,
        from_neg_one_real=True,
        from_neg_one_fake=True,
    )
    print(f"==> [{name}] FID: {fid_val:.2f} | Latency(64): {lat_m['mean_ms']:.3f} ms")

    return {
        "latent_dim": latent_dim,
        "train_time_per_epoch_s": sum(epoch_times) / len(epoch_times),
        "latency_ms_batch64": lat_m["mean_ms"],
        "test_mse": None,
        "test_psnr": None,
        "test_ssim": None,
        "fid": fid_val,
        "g_loss_history": g_losses,
        "d_loss_history": d_losses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = get_device()
    train_loader, _, test_loader = load_mnist(batch_size=128, seed=42, normalize_neg_one=True)

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    res_dir = os.path.join(args.save_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    all_metrics = {}

    # 1. Vanilla GAN (MLP)
    set_seed(42)
    v_gen = VanillaGenerator(latent_dim=100).to(device)
    v_disc = VanillaDiscriminator().to(device)
    all_metrics["vanilla_gan"] = train_single_gan(
        name="vanilla_gan",
        gen=v_gen,
        disc=v_disc,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=2e-4,
        betas=(0.5, 0.999),
        ckpt_dir=ckpt_dir,
    )

    # 2. DCGAN (Convolutional)
    set_seed(42)
    dc_gen = DCGANGenerator(latent_dim=100).to(device)
    dc_disc = DCGANDiscriminator().to(device)
    dc_gen.apply(weights_init_dcgan)
    dc_disc.apply(weights_init_dcgan)
    all_metrics["dcgan"] = train_single_gan(
        name="dcgan",
        gen=dc_gen,
        disc=dc_disc,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=2e-4,
        betas=(0.5, 0.999),
        ckpt_dir=ckpt_dir,
    )

    out_path = os.path.join(res_dir, "gan_family_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved GAN family checkpoints and metrics to: {out_path}")


if __name__ == "__main__":
    main()