"""Run all 5 stages of the MLP Classifier limitation proof (Section 2)."""

import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import load_mnist
from src.metrics import evaluate_reconstruction
from src.models.mlp_classifier import FrozenLatentDecoder, MLPClassifier
from src.utils import get_device, set_seed


def train_classifier(model, train_loader, val_loader, device, epochs=15, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader.dataset)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[MLP Classifier] Epoch {epoch:02d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")


@torch.no_grad()
def evaluate_accuracy(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def train_frozen_decoder(classifier, decoder, train_loader, device, epochs=20, lr=1e-3):
    # Tahap 4: Freeze all classifier parameters
    for param in classifier.parameters():
        param.requires_grad = False
    classifier.eval()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        decoder.train()
        total_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            with torch.no_grad():
                z_mlp = classifier.extract_latent(x)
            optimizer.zero_grad()
            recon = decoder(z_mlp)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[Frozen Decoder] Epoch {epoch:02d}/{epochs} | Train MSE: {avg_loss:.5f}")


def save_preview_figure(real_imgs, class_proj, latent_proj, frozen_recon, prior_samples, save_path):
    """Save quick visual verification of Stages 2, 3, 4, and 5."""
    rows = [
        ("Ground Truth (Test Set)", real_imgs[:8]),
        ("Stage 2: W^T from One-Hot Class (0-7)", class_proj[:8]),
        ("Stage 3: W^T from Real z_MLP", latent_proj[:8]),
        ("Stage 4: Frozen Latent Decoder", frozen_recon[:8]),
        ("Stage 5: Random Prior Sampling z_rand", prior_samples[:8]),
    ]

    fig, axes = plt.subplots(len(rows), 8, figsize=(12, 7.5))
    for r_idx, (title, imgs) in enumerate(rows):
        imgs_np = imgs.detach().cpu().numpy()
        for c_idx in range(8):
            ax = axes[r_idx, c_idx]
            ax.imshow(imgs_np[c_idx, 0], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if c_idx == 0:
                ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--cls_epochs", type=int, default=15)
    parser.add_argument("--dec_epochs", type=int, default=20)
    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    ckpt_dir = os.path.join(args.save_dir, "checkpoints")
    res_dir = os.path.join(args.save_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    train_loader, val_loader, test_loader = load_mnist(batch_size=128, seed=42, normalize_neg_one=False)

    # --- Tahap 1: Train MLP Classifier ---
    classifier = MLPClassifier().to(device)
    train_classifier(classifier, train_loader, val_loader, device, epochs=args.cls_epochs)
    test_acc = evaluate_accuracy(classifier, test_loader, device)
    print(f"==> [Stage 1] MLP Classifier Test Accuracy: {test_acc:.2f}% (Target >= 88%)")
    torch.save(classifier.state_dict(), os.path.join(ckpt_dir, "mlp_classifier.pt"))

    # --- Tahap 4: Train Frozen Latent Decoder ---
    decoder = FrozenLatentDecoder().to(device)
    train_frozen_decoder(classifier, decoder, train_loader, device, epochs=args.dec_epochs)
    torch.save(decoder.state_dict(), os.path.join(ckpt_dir, "mlp_frozen_decoder.pt"))

    # Evaluate Frozen Latent Decoder on Test Set
    decoder.eval()
    recon_metrics = evaluate_reconstruction(
        lambda x: decoder(classifier.extract_latent(x)),
        test_loader,
        device,
        from_neg_one=False,
    )
    print(f"==> [Stage 4] Frozen Latent Decoder Test Metrics: {recon_metrics}")

    # --- Tahap 2, 3, & 5: Projections and Prior Sampling ---
    sample_imgs, _ = next(iter(test_loader))
    sample_imgs = sample_imgs[:8].to(device)

    with torch.no_grad():
        # Stage 2: One-hot class vectors 0..9
        y_onehot = F.one_hot(torch.arange(10, device=device), num_classes=10).float()
        class_proj = classifier.transpose_from_class(y_onehot)

        # Stage 3: Transpose projection from real test z_MLP
        z_sample = classifier.extract_latent(sample_imgs)
        latent_proj = classifier.transpose_from_latent(z_sample)

        # Stage 4: Frozen decoder reconstruction
        frozen_recon = decoder(z_sample)

        # Stage 5: Compute mu_z, sigma_z on entire Test Set & draw z_rand ~ N(mu_z, diag(sigma_z^2))
        all_z = []
        for x, _ in test_loader:
            all_z.append(classifier.extract_latent(x.to(device)))
        all_z = torch.cat(all_z, dim=0)
        mu_z = all_z.mean(dim=0)
        sigma_z = all_z.std(dim=0)

        z_rand = mu_z + sigma_z * torch.randn(16, mu_z.size(0), device=device)
        prior_samples = decoder(z_rand)

    # Save metrics & preview image
    results = {
        "mlp_test_accuracy_pct": test_acc,
        "frozen_decoder_test_mse": recon_metrics["mse"],
        "frozen_decoder_test_psnr": recon_metrics["psnr"],
        "frozen_decoder_test_ssim": recon_metrics["ssim"],
        "z_mlp_mean_norm": float(mu_z.norm().item()),
        "z_mlp_std_mean": float(sigma_z.mean().item()),
    }
    with open(os.path.join(res_dir, "mlp_proof_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    preview_path = os.path.join(res_dir, "mlp_proof_preview.png")
    save_preview_figure(sample_imgs, class_proj, latent_proj, frozen_recon, prior_samples, preview_path)
    print(f"Saved checkpoints and preview to: {args.save_dir}")


if __name__ == "__main__":
    main()