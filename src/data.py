"""MNIST dataset loading with an 80/10/10 train-validation-test split."""

from torch.utils.data import DataLoader


def load_mnist(*args, **kwargs) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create MNIST train, validation, and test data loaders."""
    raise NotImplementedError("MNIST loader has not been implemented yet")
