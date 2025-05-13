import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_mnist(batch_size=64, download=True):
    """
    Load the MNIST dataset and return the DataLoader
    
    Args:
        batch_size (int): Batch size for training
        download (bool): Whether to download the dataset if not present
        
    Returns:
        train_loader (DataLoader): DataLoader for the MNIST training set
    """
    print("Loading MNIST dataset...")
    
    # Standard transformation for MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load the training dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=download, 
        transform=transform
    )
    
    # Create the DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    print(f"Dataset loaded: {len(train_dataset)} samples, {len(train_loader)} batches")
    return train_loader 