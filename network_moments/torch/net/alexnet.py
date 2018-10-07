from torch import nn
from ..utils import Flatten
from .base import Classifier
from torchvision.models import alexnet


__all__ = ['AlexNet']


class AlexNet(Classifier):
    min_size = 21

    def __init__(self, num_classes=1000, input_size=224,
                 input_mean=(0.485, 0.456, 0.406),
                 input_std=(0.229, 0.224, 0.225)):
        # adjusting the architecture based on the input size
        type(self).check_min_size(input_size)
        s, p, k = 2, 2, 3
        size = (input_size - 31) // 32
        if input_size < 64:
            s, p, k = (1 if input_size < 29 else 2), 5, 2
            size = (input_size + 19) // 40

        channles = len(input_mean)
        super().__init__(
            [
                nn.Conv2d(channles, 64, kernel_size=11, stride=4, padding=p),
                nn.MaxPool2d(kernel_size=k, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.MaxPool2d(kernel_size=k, stride=s),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=k, stride=2),
                nn.ReLU(inplace=True),
            ], [
                Flatten(),
                nn.Dropout(),
                nn.Linear(256 * size * size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            ],
            num_classes=num_classes, input_size=input_size,
            input_mean=input_mean, input_std=input_std,
        )

    @classmethod
    def pretrained_state_dict(cls):
        weights = alexnet(pretrained=True).state_dict().values()
        return dict(zip(cls().state_dict().keys(), weights))
