from torch import nn
from ..utils import Flatten
from .base import Classifier
from torchvision.models import vgg16


__all__ = ['VGG16']


class VGG16(Classifier):
    min_size = 16

    def __init__(self, num_classes=1000, input_size=224,
                 input_mean=(0.485, 0.456, 0.406),
                 input_std=(0.229, 0.224, 0.225),
                 batch_norm=True):
        min_size = type(self).check_min_size(input_size)

        def block(m, n, pool=True):
            conv = [nn.Conv2d(m, n, kernel_size=3, padding=1)]
            bn = [nn.BatchNorm2d(n)] if batch_norm else []
            pool = [nn.MaxPool2d(kernel_size=2, stride=2)] if pool else []
            relu = [nn.ReLU(inplace=True)]
            return conv + bn + pool + relu

        channles = len(input_mean)
        factor = 1 if input_size < 2 * min_size else 2
        size = input_size // (min_size * factor)
        super().__init__(
            [
                *block(channles, 64, pool=False),
                *block(64, 64, pool=True),
                *block(64, 128, pool=False),
                *block(128, 128, pool=True),
                *block(128, 256, pool=False),
                *block(256, 256, pool=False),
                *block(256, 256, pool=True),
                *block(256, 512, pool=False),
                *block(512, 512, pool=False),
                *block(512, 512, pool=True),
                *block(512, 512, pool=False),
                *block(512, 512, pool=False),
                *block(512, 512, pool=factor == 2),
            ], [
                Flatten(),
                nn.Linear(512 * size * size, 4096), nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, num_classes),
            ],
            num_classes=num_classes, input_size=input_size,
            input_mean=input_mean, input_std=input_std,
        )

    @classmethod
    def pretrained_state_dict(cls):
        weights = vgg16(pretrained=True).state_dict().values()
        return dict(zip(cls().state_dict().keys(), weights))
