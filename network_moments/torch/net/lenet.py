from torch import nn
from ..utils import Flatten
from .base import Classifier


__all__ = ['LeNet']


class LeNet(Classifier):
    min_size = 4

    def __init__(self, num_classes=10, input_size=28,
                 input_mean=(0.5,), input_std=(1 / 255,)):
        channles = len(input_mean)
        size = input_size // type(self).check_min_size(input_size)
        super().__init__(
            [
                nn.Conv2d(channles, 32, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
            ], [
                Flatten(),
                nn.Linear(64 * size * size, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, num_classes),
            ],
            num_classes=num_classes, input_size=input_size,
            input_mean=input_mean, input_std=input_std,
        )
