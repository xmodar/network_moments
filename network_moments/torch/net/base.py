import torch
from math import sqrt
from itertools import islice
from torchvision import transforms
from collections import OrderedDict
from ..utils.stats import trapz
from ..utils.rand import RNG, ring
from ..utils import Flatten, verbosify


__all__ = ['Sequential', 'Classifier']


class Sequential(torch.nn.Sequential):
    def forward(self, x, layers=-1):
        '''Forward pass through the layers of the network.

        Args:
            x: Input tensor.
            layers: The index of the layer to compute its output
                or a list of layers to compose their outputs.

        Returns:
            The output of the layers at `x`.
        '''
        if isinstance(layers, int):
            if layers == -1:
                return super().forward(x)
            layers = self[:layers % len(self) + 1]
        for layer in layers:
            x = layer(x)
        return x

    @staticmethod
    def encapsulate(*args):
        '''Create a squential function from a list of layers.

        Args:
            *args: The input can be one or more lists of layers.
                If multiple lists were provided,
                a list of functions will be returned.

        Returns:
            A function that sequantially feed the output of each layer as
            input to the following layer. A list of such functions will be
            retured if a list of inputs were given where zero-lengths lists
            will be ingnored.
        '''
        if len(args) > 1:
            return list(Sequential.encapsulate(x) for x in args if len(x) > 0)
        layers = args[0]
        if len(layers) == 1:
            return layers[0]

        def f(x):
            for layer in layers:
                x = layer(x)
            return x
        f.__doc__ = '\n'.join([str(y) for y in layers])
        f.layers = layers
        return f

    @staticmethod
    def split_layers(layers, separators=(torch.nn.ReLU, Flatten)):
        '''Encapsulation while exposing only layers of specific types.

        Args:
            layers: List of layers.
            separators: A list of specific types to expose.

        Returns:
            A list of splitted layers.
        '''
        out_layers = []
        bunch = []
        for layer in layers:
            if any(isinstance(layer, x) for x in separators):
                if len(bunch) > 0:
                    out_layers.append(Sequential.encapsulate(bunch))
                out_layers.append(layer)
                bunch = []
            else:
                bunch.append(layer)
        if len(bunch) > 0:
            out_layers.append(Sequential.encapsulate(bunch))
        return out_layers

    def __iter__(self):
        return iter(self._modules.values())

    def __getitem__(self, index):
        values = self._modules.values()
        if isinstance(index, slice):
            start = (None if index.start is None
                     else index.start % len(values))
            stop = (None if index.stop is None
                    else index.stop % len(values))
            return list(islice(values, start, stop, index.step))
        return next(islice(values, index, None))


class Classifier(Sequential):
    min_size = 1

    def __init__(self, features, classifier, num_classes,
                 input_size, input_mean, input_std):
        '''Initializes the classifier.

        Args:
            features: The layers list of the features extractor.
            classifiers: The layers list of the features classifier.
            num_classes: Number of output classes of the classifier.
            input_size: The input size of the classifier.
            input_mean: The mean of the channels in the input image.
            input_std: The standard deviations of the input channels.
        '''
        layers = features + classifier
        as_dict = len(layers) > 0 and isinstance(layers[0], tuple)
        if as_dict:
            super().__init__(OrderedDict(layers))
        else:
            super().__init__(*layers)
        self.features_layer = len(features)
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_mean = input_mean
        self.input_std = input_std

    @classmethod
    def check_min_size(cls, input_size):
        '''Assert that the input_size is at least as big as min_size.'''
        min_size = cls.min_size
        if input_size < min_size:
            msg = f'(must be at least {min_size})'
            raise ValueError(f'`input_size`={input_size} is too small {msg}')
        return min_size

    @staticmethod
    def basic_transform(size=tuple(), pil=tuple(), tns=tuple(), mean=0, std=1):
        '''Basic transformation function for classifier input.

        The following transformations are carried in order:
        [`size`, `pil`, `to_tensor`, `trans`, `normalize`]
        `to_tensor` converts `PIL` image to `torch.tensor`.
        `normalize` subtracts `mean` divides by `std`.

        Args:
            size: List of resizing operations or the integer size.
            pil: List of additional image transformation functions.
            tns: List of additional tensor transformations.
            mean: The mean to use for normalization.
            std: The standard deviation to use for normalization.

        Returns:
            The transformation function.
        '''
        to_tensor = (transforms.ToTensor(),)
        normalize = tuple() if (mean == 0 and std == 1) else (
            transforms.Normalize(mean=mean, std=std),)
        if isinstance(size, int):
            size = (
                transforms.Resize(size),
                transforms.CenterCrop(size),
            )
        trans = size + tuple(pil) + to_tensor + tuple(tns) + normalize
        return transforms.Compose(trans)

    def default_transform(self, pil=tuple(), tns=tuple(), resize=True):
        '''Default transformation function for classifier input.

        The following transformations are carried in order:
        [`resize`, `pil`, `to_tensor`, `trans`, `normalize`]
        where `resize` is center crop after rescale to `self.input_size`.
        `to_tensor` converts `PIL` image to `torch.tensor`.
        `normalize` subtracts `self.input_mean` divides by `self.input_std`.

        Args:
            pil: List of additional image transformation functions.
            tns: List of additional tensor transformations.
            resize: Whether to resize the image to fit self.input_size.

        Returns:
            The transformation function.
        '''
        return self.basic_transform(
            pil=pil, tns=tns,
            mean=self.input_mean, std=self.input_std,
            size=self.input_size if resize else tuple())

    def input_range(self):
        return sum(float(1 / v) for v in self.input_std) / len(self.input_std)

    def extract_features(self, x):
        return self.eval().forward(x, layers=self[:self.features_layer])

    def classify(self, x):
        return self.eval().forward(x, layers=self[self.features_layer:])

    def accuracy(self, loader, device):
        '''Computes the accuracy of the classifier.

        Args:
            loader: A torch.utils.data.DataLoader.
            device: The device to do the forward passes on.

        Returns:
            The computed accuracy of the model.
        '''
        self.to(device).eval()
        with torch.no_grad():
            total = count = 0
            for images, labels in verbosify(loader, leave=False):
                total += images.size(0)
                pred = self.forward(images.to(device)).argmax(-1)
                count += pred.eq(labels.to(device)).sum()
            return count.item() / total

    @torch.no_grad()
    def gaussian_robustness(self, loader, device, sigmas_range=(0, 0.5, 30)):
        '''Compute the robustness in the presence of Gaussian noise.

        We first define the accuracy at a give input noise level sigma
        as a function rho(sigma). Then, we define the robustness to be the
        area under the curve of rho when sigma is in the given range.
        Note: rho(sigma) is independent of the ground truth labels
            and the noise is generated using `utils.rand.ring()`.

        Args:
            loader: A torch.utils.data.DataLoader without shuffling.
            device: The device to do the forward passes on.
            sigmas_range: (min_sigma, max_sigma, num_sigmas)
                The noise levels are `linspace(*sigmas_range)`.
                The generated noise will be multiplied by `self.input_range()`.

        Returns:
            (robustness: The robustness of the classifier under Gaussian noise,
             (sigmas: The noise levels, accuracies: the accuracies))
        '''
        self.to(device).eval()

        # compute the output predictions for the clean images
        count = 0
        clean_labels = []
        with torch.no_grad():
            for images, _ in loader:
                labels = self.forward(images.to(device)).argmax(dim=-1)
                clean_labels.append(labels)
                count += images.size(0)

        # get the noise levels (sigmas)
        input_range = self.input_range()
        sigmas = torch.linspace(*sigmas_range, device=device)
        accuracies = torch.zeros(sigmas.numel(), device=device)
        kwargs = {
            'device': device,
            'dtype': images.dtype,
            'size': images[0, ...].size(),
            'tolerance': float((sigmas[1] - sigmas[0]) / 2),
        }

        # seeding the device to get deterministic output
        with RNG(seed=0, devices=[device]):
            for i, sigma in enumerate(verbosify(sigmas)):
                sigma = float(sigma * input_range)
                for (images, _), labels in zip(loader, clean_labels):
                    noise = ring(batch=images.size(0), sigma=sigma, **kwargs)
                    out = self.forward(noise.add_(images.to(device)))
                    accuracies[i] += int(out.argmax(dim=-1).eq(labels).sum())

        accuracies /= count
        robustness = trapz(accuracies, x=sigmas) / (sigmas[-1] - sigmas[0])
        return robustness, (sigmas, accuracies)
