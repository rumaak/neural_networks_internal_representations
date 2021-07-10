from torchvision import datasets
from torchvision.transforms import ToTensor

def load_FashionMNIST():
    """Load FashionMNIST dataset."""
    training_data = datasets.FashionMNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data

def load_MNIST():
    """Load MNIST dataset."""
    training_data = datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data
