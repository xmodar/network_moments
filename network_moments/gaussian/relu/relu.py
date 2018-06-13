import math
import torch
from ...utils import diagonal, outer


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

    Returns:
        Output mean of ReLU for general Gaussian input (Batch, Size).
    '''
    std = variance if std else torch.sqrt(variance)
    temp1 = std / math.sqrt(2.0 * math.pi)
    if not torch.is_tensor(mean) and mean == 0.0:
        return temp1  # efficient computation when mean is zeros
    u = mean / (math.sqrt(2.0) * std)
    temp2 = 0.5 * mean * (1.0 + torch.erf(u))
    return temp1 * torch.exp(-u ** 2.0) + temp2


def zero_mean_correlation(covariance, stability=0.0, _std=None):
    '''Output correlation of ReLU for zero-mean Gaussian input.

    f(x) = max(x, 0).

    Args:
        covariance: Input covariance matrix (Size, Size).
        stability: For accurate results this should be zero
            if used in training, use a value like 1e-4 for stability.

    Returns:
        Output correlation of ReLU for zero-mean Gaussian input (Size, Size).
    '''
    S = outer(torch.sqrt(diagonal(covariance, 0, -2, -1))
              if _std is None else _std)  # use precomputed _std if provided
    V = (covariance / S).clamp_(stability - 1.0, 1.0 - stability)
    temp1 = covariance * torch.asin(V)
    temp2 = S * torch.sqrt(1.0 - (V ** 2.0))
    return (temp1 + temp2) / (2.0 * math.pi) + covariance / 4.0


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
    std = torch.sqrt(diagonal(covariance, 0, -2, -1))
    mu = mean(0, std, std=True)
    corr = zero_mean_correlation(covariance, stability, _std=std)
    return corr - outer(mu)
