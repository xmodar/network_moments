from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T

__all__ = ['get_mnist', 'get_mnist_loader', 'get_lenet', 'get_mnist_lenet']


def get_mnist(data_path=None):
    """Get the MNIST train/test split."""
    if data_path is None:
        data_path = Path.home() / '.torch/datasets/MNIST'
    r = datasets.MNIST(data_path, train=True, transform=T.ToTensor())
    s = datasets.MNIST(data_path, train=False, transform=T.ToTensor())
    return r, s


def get_mnist_loader(data_path=None, train=500, test=1000, cuda=None):
    """Get dataloaders for the MNIST train/test split."""
    train_set, test_set = get_mnist(data_path)
    cuda = torch.cuda.is_available() if cuda is None else cuda
    r = DataLoader(train_set, batch_size=train, shuffle=True, pin_memory=cuda)
    s = DataLoader(test_set, batch_size=test, shuffle=False, pin_memory=cuda)
    return r, s


def get_lenet(checkpoint=None, input_size=28, channles=1, num_classes=10):
    """Build a sequential LeNet model."""
    size = input_size // 4
    lenet = nn.Sequential(
        nn.Conv2d(channles, 32, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=5, padding=2),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
        nn.Flatten(),
        nn.Linear(64 * size * size, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, num_classes),
    )
    if checkpoint is not None:
        lenet.load_state_dict(torch.load(checkpoint, 'cpu'))
    return lenet


def train_classifier(epochs, model, train_loader, test_loader, device):
    """Train a classifier model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(1, epochs + 1):
        model.train(True)
        train_accuracy = 0
        for xs, ys in train_loader:
            xs, ys = xs.to(device), ys.to(device)
            logits = model(xs)
            loss = F.cross_entropy(logits, ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_accuracy += (logits.argmax(1) == ys).sum().item()
        train_accuracy *= 100 / len(train_loader.dataset)
        with torch.no_grad():
            model.train(False)
            test_accuracy = 0
            for xs, ys in test_loader:
                xs, ys = xs.to(device), ys.to(device)
                test_accuracy += (model(xs).argmax(1) == ys).sum().item()
            test_accuracy *= 100 / len(test_loader.dataset)
        print(f'{epoch} Train {train_accuracy:.2f}% Test {test_accuracy:.2f}%')
    return model


def get_mnist_lenet(checkpoint='./mnist_lenet.pt', device=None, epochs=3):
    """Get LeNet trained on MNIST from a checkpoint or train from scratch."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if Path(checkpoint).exists():
        model = get_lenet(checkpoint).to(device)
    else:
        cuda = device.type == 'cuda'
        model = get_lenet().to(device)
        train_loader, test_loader = get_mnist_loader(cuda=cuda)
        train_classifier(epochs, model, train_loader, test_loader, device)
        torch.save(model.state_dict(), checkpoint)
    return model
