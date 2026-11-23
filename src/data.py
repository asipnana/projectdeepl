"""MNIST dataset loading with an 80/10/10 train-validation-test split."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split
from torchvision import datasets, transforms


def load_mnist(
    data_dir: str = "./data",
    batch_size: int = 128,
    seed: int = 42,
    normalize_neg_one: bool = False,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create standardized MNIST train (80%), validation (10%), and test (10%) loaders.
    normalize_neg_one=False -> [0, 1] (MLP, Vanilla AE, Sparse AE, VAE)
    normalize_neg_one=True  -> [-1, 1] (Vanilla GAN, DCGAN, DDPM)
    """
    transform_list = [transforms.ToTensor()]
    if normalize_neg_one:
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    transform = transforms.Compose(transform_list)

    train_raw = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_raw = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    full_dataset = ConcatDataset([train_raw, test_raw])

    total_size = len(full_dataset)  # 70,000
    train_size = int(0.8 * total_size)  # 56,000
    val_size = int(0.1 * total_size)    # 7,000
    test_size = total_size - train_size - val_size  # 7,000

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_l, val_l, test_l = load_mnist()
    images, labels = next(iter(train_l))
    print(f"Train: {len(train_l.dataset)} | Val: {len(val_l.dataset)} | Test: {len(test_l.dataset)}")
    print(f"Batch shape: {images.shape} | Min: {images.min():.2f} | Max: {images.max():.2f}")