"""Generate all 5 mandatory high-resolution figures (300 DPI) and summary metrics table."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.data import load_mnist
from src.metrics import rescale_to_unit
from src.models.dcgan import DCGANGenerator
from src.models.ddpm import DDPM
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier
from src.models.sparse_ae import SparseAE
from src.models.vae import VAE
from src.models.vanilla_ae import VanillaAE
from src.models.vanilla_gan import VanillaGenerator
from src.utils import get_device, set_seed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./outputs")
    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    res_dir = os.path.join(args.save_dir, "results")
    fig_dir = os.path.join(res_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    _, _, test_loader = load_mnist(batch_size=128, seed=42, normalize_neg_one=False)
    test_imgs, test_labels = next(iter(test_loader))
    test_imgs = test_imgs.to(device)

    # Load all trained models
    mlp = MLPClassifier().to(device)
    mlp.load_state_dict(torch.load(os.path.join(ckpt_dir, "mlp_classifier.pt"), map_location=device))
    mlp.eval()

    frozen_dec = FrozenLatentDecoder().to(device)
    frozen_dec.load_state_dict(torch.load(os.path.join(ckpt_dir, "mlp_frozen_decoder.pt"), map_location=device))
    frozen_dec.eval()

    vanilla_ae = VanillaAE(latent_dim=64).to(device)
    vanilla_ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "vanilla_ae.pt"), map_location=device))
    vanilla_ae.eval()

    sparse_ae = SparseAE(latent_dim=64).to(device)
    sparse_ae.load_state_dict(torch.load(os.path.join(ckpt_dir, "sparse_ae.pt"), map_location=device))
    sparse_ae.eval()

    vae = VAE(latent_dim=64).to(device)
    vae.load_state_dict(torch.load(os.path.join(ckpt_dir, "vae.pt"), map_location=device))
    vae.eval()

    v_gan = VanillaGenerator(latent_dim=100).to(device)
    v_gan.load_state_dict(torch.load(os.path.join(ckpt_dir, "vanilla_gan_gen.pt"), map_location=device))
    v_gan.eval()

    dcgan = DCGANGenerator(latent_dim=100).to(device)
    dcgan.load_state_dict(torch.load(os.path.join(ckpt_dir, "dcgan_gen.pt"), map_location=device))
    dcgan.eval()

    ddpm = DDPM(timesteps=200).to(device)
    ddpm.load_state_dict(torch.load(os.path.join(ckpt_dir, "ddpm.pt"), map_location=device))
    ddpm.eval()

    # =========================================================================
    # FIGUR 1: Grid Rekonstruksi Side-by-Side (8 sampel Test Set)
    # =========================================================================
    x8 = test_imgs[:8]
    r_vae_ae, _ = vanilla_ae(x8)
    r_sae, _ = sparse_ae(x8)
    r_vae, _, _ = vae(x8)

    rows_f1 = [
        ("Ground Truth (Test Set)", x8),
        ("Vanilla AE Reconstruction", r_vae_ae),
        ("Sparse AE Reconstruction", r_sae),
        ("VAE Reconstruction", r_vae),
    ]
    fig, axes = plt.subplots(4, 8, figsize=(12, 6.2))
    for r, (title, imgs) in enumerate(rows_f1):
        arr = imgs.cpu().numpy()
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(arr[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=5)
    plt.suptitle("Figur 1: Perbandingan Rekonstruksi Citra Test Set (Vanilla AE vs Sparse AE vs VAE)", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figur1_rekonstruksi.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figur 1")

    # =========================================================================
    # FIGUR 2: Grid Random Generative Sampling (4x4 per model)
    # =========================================================================
    set_seed(42)
    s_vae = vae.sample(16, device).cpu()
    s_vgan = rescale_to_unit(v_gan.sample(16, device), from_neg_one=True).cpu()
    s_dcgan = rescale_to_unit(dcgan.sample(16, device), from_neg_one=True).cpu()
    s_ddpm, ddpm_traj = ddpm.sample(16, device, return_trajectory=True)
    s_ddpm = rescale_to_unit(s_ddpm, from_neg_one=True).cpu()

    panels_f2 = [
        ("Panel A: VAE (Prior z ~ N(0,I))", s_vae),
        ("Panel B: Vanilla GAN (MLP)", s_vgan),
        ("Panel C: DCGAN (Convolutional)", s_dcgan),
        ("Panel D: DDPM (Reverse Denoising)", s_ddpm),
    ]
    fig = plt.figure(figsize=(12, 11))
    outer = fig.add_gridspec(2, 2, wspace=0.15, hspace=0.22)
    for idx, (title, samples) in enumerate(panels_f2):
        inner = outer[idx // 2, idx % 2].subgridspec(4, 4, wspace=0.05, hspace=0.05)
        ax_title = fig.add_subplot(outer[idx // 2, idx % 2])
        ax_title.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax_title.axis("off")
        for i in range(16):
            ax = fig.add_subplot(inner[i // 4, i % 4])
            ax.imshow(samples[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
    plt.suptitle("Figur 2: Grid Random Generative Sampling (4x4 Sampel Sintetis)", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(fig_dir, "figur2_generative_sampling.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figur 2")

    # =========================================================================
    # FIGUR 3: Bukti Empiris Batasan MLP Classifier (vs Vanilla AE)
    # =========================================================================
    y_oh = F.one_hot(torch.arange(8, device=device), num_classes=10).float()
    proj_class = mlp.transpose_from_class(y_oh)
    z_mlp_8 = mlp.extract_latent(x8)
    proj_latent = mlp.transpose_from_latent(z_mlp_8)
    frozen_rec_8 = frozen_dec(z_mlp_8)

    all_z = torch.cat([mlp.extract_latent(x.to(device)) for x, _ in test_loader], dim=0)
    mu_z, std_z = all_z.mean(dim=0), all_z.std(dim=0)
    z_rand = mu_z + std_z * torch.randn(8, mu_z.size(0), device=device)
    prior_mlp = frozen_dec(z_rand)

    rows_f3 = [
        ("1. Ground Truth (Test Set)", x8),
        ("2a. W^T Projection from One-Hot Class (0..7)", proj_class),
        ("2b. W^T Projection from Real z_MLP", proj_latent),
        ("3a. Frozen Latent Decoder (z_MLP)", frozen_rec_8),
        ("3b. Vanilla AE Reconstruction (z_AE, dim=64)", r_vae_ae),
        ("4. Random Prior Sampling z_rand ~ N(mu_z, sigma_z^2)", prior_mlp),
    ]
    fig, axes = plt.subplots(6, 8, figsize=(12, 9.2))
    for r, (title, imgs) in enumerate(rows_f3):
        arr = imgs.cpu().numpy()
        for c in range(8):
            ax = axes[r, c]
            ax.imshow(arr[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c == 0:
                ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=4)
    plt.suptitle("Figur 3: Bukti Empiris Batasan Bobot (W^T) & Ruang Laten (z_MLP) MLP Classifier", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figur3_mlp_limitation_proof.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figur 3")

    # =========================================================================
    # FIGUR 4: Linear Latent Space Interpolation (alpha in [0.0..1.0])
    # =========================================================================
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    idx_a = (test_labels == 0).nonzero(as_tuple=True)[0][0].item()
    idx_b = (test_labels == 7).nonzero(as_tuple=True)[0][0].item()
    img_a, img_b = test_imgs[idx_a:idx_a + 1], test_imgs[idx_b:idx_b + 1]

    def interp_list(z_a, z_b, dec_fn, from_neg_one=False):
        out = []
        for a in alphas:
            z = (1.0 - a) * z_a + a * z_b
            im = dec_fn(z)
            if from_neg_one:
                im = rescale_to_unit(im, from_neg_one=True)
            out.append(im[0, 0].cpu().numpy())
        return out

    set_seed(123)
    z_gan_a = torch.randn(1, 100, device=device)
    z_gan_b = torch.randn(1, 100, device=device)

    interp_rows = [
        ("Vanilla AE (0 -> 7)", interp_list(vanilla_ae.encode(img_a), vanilla_ae.encode(img_b), vanilla_ae.decode)),
        ("Sparse AE (0 -> 7)", interp_list(sparse_ae.encode(img_a), sparse_ae.encode(img_b), sparse_ae.decode)),
        ("VAE (0 -> 7)", interp_list(vae.encode(img_a)[0], vae.encode(img_b)[0], vae.decode)),
        ("Vanilla GAN (z_A -> z_B)", interp_list(z_gan_a, z_gan_b, v_gan, from_neg_one=True)),
        ("DCGAN (z_A -> z_B)", interp_list(z_gan_a, z_gan_b, dcgan, from_neg_one=True)),
    ]
    fig, axes = plt.subplots(5, len(alphas), figsize=(10.5, 8.5))
    for r, (label, imgs_row) in enumerate(interp_rows):
        for c, a in enumerate(alphas):
            ax = axes[r, c]
            ax.imshow(imgs_row[c], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"alpha = {a:.1f}", fontsize=10, fontweight="bold")
            if c == 0:
                ax.text(-5, 14, label, va="center", ha="right", fontsize=9.5, fontweight="bold")
    plt.suptitle("Figur 4: Linear Latent Space Interpolation Antarmodel", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figur4_latent_interpolation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figur 4")

    # =========================================================================
    # FIGUR 5: Training Loss Curves (6 Models) & DDPM Denoising Trajectory
    # =========================================================================
    with open(os.path.join(res_dir, "ae_family_metrics.json")) as f:
        ae_m = json.load(f)
    with open(os.path.join(res_dir, "gan_family_metrics.json")) as f:
        gan_m = json.load(f)
    with open(os.path.join(res_dir, "ddpm_metrics.json")) as f:
        ddpm_m = json.load(f)

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.35, wspace=0.28)

    # Plot 1: Vanilla AE & Sparse AE Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ae_m["vanilla_ae"]["loss_history"], label="Vanilla AE (MSE)", lw=2)
    ax1.plot(ae_m["sparse_ae"]["loss_history"], label="Sparse AE (MSE+L1)", lw=2)
    ax1.set_title("Vanilla AE vs Sparse AE Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.grid(alpha=0.3)

    # Plot 2: VAE ELBO Loss
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ae_m["vae"]["loss_history"], color="purple", lw=2)
    ax2.set_title("VAE Training Loss (ELBO)", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("BCE + KLD"); ax2.grid(alpha=0.3)

    # Plot 3: DDPM Noise Prediction Loss
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ddpm_m["ddpm"]["loss_history"], color="teal", lw=2)
    ax3.set_title("DDPM Noise Prediction Loss (MSE)", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Noise MSE"); ax3.grid(alpha=0.3)

    # Plot 4: Vanilla GAN G/D Loss
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(gan_m["vanilla_gan"]["g_loss_history"], label="G Loss", lw=2)
    ax4.plot(gan_m["vanilla_gan"]["d_loss_history"], label="D Loss", lw=2)
    ax4.set_title("Vanilla GAN (MLP) G vs D Loss", fontweight="bold")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("BCE Loss"); ax4.legend(); ax4.grid(alpha=0.3)

    # Plot 5: DCGAN G/D Loss
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(gan_m["dcgan"]["g_loss_history"], label="G Loss", lw=2)
    ax5.plot(gan_m["dcgan"]["d_loss_history"], label="D Loss", lw=2)
    ax5.set_title("DCGAN (Conv) G vs D Loss", fontweight="bold")
    ax5.set_xlabel("Epoch"); ax5.set_ylabel("BCE Loss"); ax5.legend(); ax5.grid(alpha=0.3)

    # Plot 6: DDPM Denoising Trajectory (x_T -> x_0)
    sub_gs = gs[1, 2].subgridspec(2, len(ddpm_traj), wspace=0.05, hspace=0.1)
    ax_traj_title = fig.add_subplot(gs[1, 2])
    ax_traj_title.set_title("DDPM Reverse Trajectory (x_T -> x_0)", fontweight="bold", pad=16)
    ax_traj_title.axis("off")
    for col, (step, imgs_t) in enumerate(ddpm_traj):
        imgs_u = rescale_to_unit(imgs_t, from_neg_one=True)
        for row in range(2):
            ax = fig.add_subplot(sub_gs[row, col])
            ax.imshow(imgs_u[row, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={step}", fontsize=8)

    plt.suptitle("Figur 5: Kurva Training Loss Keenam Model & Sekuens Reverse Diffusion DDPM", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(fig_dir, "figur5_loss_dan_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figur 5")

    # =========================================================================
    # PRINT SUMMARY TABLE FOR BAGIAN IV
    # =========================================================================
    all_m = {**ae_m, **gan_m, **ddpm_m}
    print("\n================ TABEL METRIK KOMPARATIF (BAGIAN IV) ================")
    print(f"{'Model':<14} | {'Latent':<6} | {'Time/Ep(s)':<10} | {'Lat(ms/64)':<10} | {'MSE':<8} | {'PSNR(dB)':<8} | {'SSIM':<6} | {'FID':<6}")
    print("-" * 88)
    for name in ["vanilla_ae", "sparse_ae", "vae", "vanilla_gan", "dcgan", "ddpm"]:
        m = all_m[name]
        mse_s = f"{m['test_mse']:.5f}" if m["test_mse"] is not None else "N/A"
        psnr_s = f"{m['test_psnr']:.2f}" if m["test_psnr"] is not None else "N/A"
        ssim_s = f"{m['test_ssim']:.4f}" if m["test_ssim"] is not None else "N/A"
        fid_s = f"{m['fid']:.2f}" if m["fid"] is not None else "N/A"
        print(f"{name:<14} | {str(m['latent_dim']):<6} | {m['train_time_per_epoch_s']:<10.2f} | {m['latency_ms_batch64']:<10.3f} | {mse_s:<8} | {psnr_s:<8} | {ssim_s:<6} | {fid_s:<6}")


if __name__ == "__main__":
    main()