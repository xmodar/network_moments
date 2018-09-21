import math
import torch
from ...utils import outer
from ...utils.stats.gaussian import normal_density


__all__ = ['moments', 'mean', 'variance', 'zero_mean_covariance']


def moments(mu, var, mean=True, variance=True, std=False):
    '''Output mean and variance of ReLU for general Gaussian input.

    f(x) = max(x, 0).

    This function is broadcast-able, so you can provide multiple
    input means with a single variance or multiple input variances
    with a single input mean or multiple input means and variances.

    Args:
        mu: Input mean of size (Batch, Size).
        var: Input variance vector (Batch, Size)
            or scalar v such that variance = v * ones(Size).
        mean: Whether to output the mean.
        variance: Whether to output the variance.
        std: Whether the provided `var` is the standard deviation.

    Returns:
        Output mean and variance of ReLU for general Gaussian input.
    '''
    sigma = var if std else torch.sqrt(var)
    mu_sigma = (torch.tensor(0, dtype=var.dtype, device=var.device)
                if mu is None else mu / sigma)
    mu = 0 if mu is None else mu
    pdf, cdf = normal_density(mu_sigma)
    zero_mean = sigma * pdf
    relu_mean = mu * cdf + zero_mean
    if not variance:
        return relu_mean
    var = var ** 2.0 if std else var
    relu_second_moment = (mu**2.0 + var) * cdf + mu * zero_mean
    relu_variance = relu_second_moment - relu_mean**2.0
    if not mean:
        return relu_variance
    return relu_mean, relu_variance


def mean(mean, variance, std=False):
    '''Output mean of ReLU for general Gaussian input.

    f(x) = max(x, 0).

    This function is broadcast-able, so you can provide multiple
    input means with a single variance or multiple input variances
    with a single input mean or multiple input means and variances.

    Args:
        mean: Input mean of size (Batch, Size).
        variance: Input variance vector (Batch, Size)
            or scalar v such that variance = v * ones(Size).
        std: Whether the provided `variance` is the standard deviation.
            This function is more efficient when `std` is True.

    Returns:
        Output mean of ReLU for general Gaussian input (Batch, Size).
    '''
    return moments(mean, variance, variance=False, std=std)


def variance(mean, variance, std=False):
    '''Output variance of ReLU for general Gaussian input.

    f(x) = max(x, 0).

    This function is broadcast-able, so you can provide multiple
    input means with a single variance or multiple input variances
    with a single input mean or multiple input means and variances.

    Args:
        mean: Input mean of size (Batch, Size).
        variance: Input variance vector (Batch, Size)
            or scalar v such that variance = v * ones(Size).
        std: Whether the provided `variance` is the standard deviation.

    Returns:
        Output variance of ReLU for general Gaussian input (Batch, Size).
    '''
    return moments(mean, variance, mean=False, std=std)


def zero_mean_covariance(covariance, stability=0.0):
    '''Output covariance of ReLU for zero-mean Gaussian input.

    f(x) = max(x, 0).

    Args:
        covariance: Input covariance matrix (Size, Size).
        stability: For accurate results this should be zero
            if used in training, use a value like 1e-4 for stability.

    Returns:
        Output covariance of ReLU for zero-mean Gaussian input (Size, Size).
    '''
    S = outer(torch.sqrt(torch.diagonal(covariance, 0, -2, -1)))
    V = (covariance / S).clamp_(stability - 1.0, 1.0 - stability)
    Q = torch.acos(-V) * V + torch.sqrt(1.0 - (V**2.0)) - 1.0
    return S * Q * (1.0 / (2.0 * math.pi))
