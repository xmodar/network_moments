import torch
from ...utils import Flatten
from ...gaussian.relu import batch_moments as relu_moments
from ...general.affine import batch_moments as affine_moments


__all__ = ['gaussian']


def gaussian(layers, mu, var, mean=True, covariance=True,
             diagonal=True, independent=True, linearize=False):
    '''Gaussian ADF.

    Computes the output mean and covariance for a sequence
    of layers given a general Gaussian input.

    Args:
        layers: A list of layers.
        mu: The mean of the input Gaussian (Batch, *size).
        var: Input covariance matrix or variance (Batch, size[, size]).
        mean: Whether to output the mean.
        covariance: Whether to output the covariance.
        diagonal: Whether to output the variance if `covariance` is True.
        independent: Whether to only compute intermediate variances
            instead of computing full covariances.
        linearize: Whether to compute the Jacobian for affine layers.

    Returns:
        The output mean and covariance of the given layers.
    '''
    last_iteration = len(layers) - 1
    for i, layer in enumerate(layers):
        last = i == last_iteration
        mn = last and not mean
        cov = last and not covariance
        dg = independent or (last and diagonal)
        if isinstance(layer, Flatten):
            mu = layer(mu)
        elif isinstance(layer, torch.nn.ReLU):
            if mn == cov:
                mu, var = relu_moments(mu, var, diagonal=dg)
            elif mn:
                mu = relu_moments(mu, var, covariance=False)
                var = None
            else:
                var = relu_moments(mu, var, mean=False, diagonal=dg)
                mu = None
        else:
            if mn == cov:
                mu, var = affine_moments(layer, mu, var, diagonal=dg,
                                         jacobian=linearize)
            elif mn:
                mu = affine_moments(layer, mu, var, covariance=False,
                                    jacobian=linearize)
                var = None
            else:
                var = affine_moments(layer, mu, var, mean=False,
                                     diagonal=dg, jacobian=linearize)
                mu = None
    if mean and not covariance:
        return mu
    if covariance and not mean:
        return var
    return mu, var
