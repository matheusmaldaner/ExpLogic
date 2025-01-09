import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.mnist_dataset import MNISTRemoveBorderTransform

# --- Dataset Selection ---
def get_dataset(dataset_name, batch_size=64, data_dir='./data', bpp=8, crop=None):
    """
    Returns the appropriate dataset based on the user input.
    Args:
        dataset_name (str): Dataset to load ('mnist', 'fashion_mnist', or 'cifar10').
        batch_size (int): Batch size for the data loader.
        data_dir (str): Directory to store/load datasets.
    
    Returns:
        DataLoader: The data loader for the selected dataset.
    """
    
    max_num   = (2**bpp-1)
    use_crop  = [transforms.Lambda(lambda x: x.crop(((x.size[0] - crop[0])/2, (x.size[1] - crop[1])/2, (x.size[0] + crop[0])/2, (x.size[1] + crop[1])/2)))] if crop else [] 
    transform = transforms.Compose(use_crop + [transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                                   transforms.ToTensor(),  # Convert images to tensor
                                   #transforms.Normalize((0.5,), (0.5,)), 
                                   transforms.Lambda(lambda x: (x * max_num + 0.5).to(torch.uint8)),  # Scale to [0, max_num] and convert to uint8
                                   transforms.Lambda(lambda x: torch.tensor([[int(i) for i in f'{int(pixel.item()):0{bpp}b}'] for pixel in x.flatten()]).flatten())]) 

    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        input_dim = 28 * 28 * bpp # MNIST/FashionMNIST images are 28x28
        out_dim = 10  # 10 classes
    elif dataset_name.lower() == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
        val_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
        input_dim = 28 * 28 * bpp  # MNIST/FashionMNIST images are 28x28
        out_dim = 10  # 10 classes
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist', of 'fashion_mnist'.")

    # DataLoader for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, input_dim, out_dim
