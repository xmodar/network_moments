import torch
from ...utils import mul_diag


def mean(mean, A, b=0.0):
    '''Output mean of Affine for general input.

    f(x) = A * x + b.

    Args:
        mean: Input mean of size (Batch, Size).
        A: Matrix of size (M, Size).
        transposed: Whether A is transposed (Size, M).

    Returns:
        Output mean of Affine for general input (Batch, M).
    '''
    if not torch.is_tensor(b) and b == 0.0:
        return mean.matmul(A.t())
    else:
        return mean.matmul(A.t()) + b


def covariance(covariance, A, variance=False):
    '''Output covariance matrix of Affine for general input.

    f(x) = A * x + b.

    Args:
        covariance: Input covariance matrix of size (Batch, Size, Size)
            or variance of size (Batch, Size).
        A: Matrix (M, Size).
        variance: Whether the input covariance is a diagonal matrix.

    Returns:
        Output covariance of Affine for general input (Batch, M, M).
    '''
    if variance:
        return mul_diag(A, covariance).matmul(A.t())
    if A.size(0) > A.size(1):
        return A.matmul(covariance.matmul(A.t()))  # a slight performance gain
    else:
        return A.matmul(covariance).matmul(A.t())


def variance(covariance, A, variance=False):
    '''Output variance of Affine for general input.

    f(x) = A * x + b.

    Args:
        covariance: Input covariance matrix of size (Batch, Size, Size)
            or variance of size (Batch, Size).
        A: Matrix (M, Size).
        variance: Whether the input covariance is a diagonal matrix.

    Returns:
        Output variance of Affine for general input (Batch, M).
    '''
    B = mul_diag(A, covariance) if variance else A.matmul(covariance)
    return torch.sum(B * A, -1)
