import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_cifar100(batch_size=64, download=True):
    """
    Load the CIFAR-100 dataset and return the DataLoader

    Args:
        batch_size (int): Batch size for training
        download (bool): Whether to download the dataset if not present

    Returns:
        train_loader (DataLoader): DataLoader for the CIFAR-100 training set
    """
    print("Loading CIFAR-100 dataset...")
    # Standard transformation for CIFAR-100
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5071, 0.4865, 0.4409),  # CIFAR-100 mean
            (0.2673, 0.2564, 0.2761)   # CIFAR-100 std
        )
    ])
    # Load the training dataset
    train_dataset = datasets.CIFAR100(
        root='./data',
        train=True,
        download=download,
        transform=transform
    )
    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    print(f"Dataset loaded: {len(train_dataset)} samples, {len(train_loader)} batches")
    return train_loader 