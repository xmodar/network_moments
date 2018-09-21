from .. import relu
from ...general import affine


__all__ = ['mean', 'special_variance', 'special_covariance']


def mean(mean, covariance, A, c1, B, c2, variance=False, precomputed=False):
    '''Output mean of Affine-ReLU-Affine for general Gaussian input.

    f(x) = B*max(A*x+c1, 0)+c2.

    This function is broadcast-able, so you can provide multiple
    input means with a single covariance or multiple input covariances
    with a single input mean or multiple input means and covariances.

    Args:
        mean: Input mean of size (Batch, Size).
        covariance: Input covariance matrix (Batch, Size, Size)
            or variance vector (Batch, Size) for diagonal covariance.
        A: The A matrix (M, Size).
        c1: The c1 vector (M).
        B: The B matrix (N, M).
        c2: The c2 vector (N).
        variance: Whether the input `covariance` is a diagonal matrix.
        precomputed: Whether the provided `covariance` is the precomputed
                variance after the first affine layer.

    Returns:
        Output mean of Affine-ReLU-Affine for Gaussian input
        with mean = `mean` and covariance matrix = `covariance`.
    '''
    a_mean = affine.mean(mean, A, c1)
    a_variance = (affine.variance(covariance, A, variance=variance)
                  if not precomputed else covariance)
    r_mean = relu.mean(a_mean, a_variance)
    return affine.mean(r_mean, B, c2)


def special_variance(covariance, A, B, variance=False, stability=0.0):
    '''Output variance of Affine-ReLU-Affine for special Gaussian input.

    f(x) = B*max(A*x+c1, 0)+c2, where c1 = -A*input_mean.

    For this specific c1, this function doesn't depend on
    neither the input mean nor the biases.

    Args:
        covariance: Input covariance matrix (Batch, Size, Size)
            or variance vector (Batch, Size) for diagonal covariance.
        A: The A matrix (M, Size).
        B: The B matrix (N, M).
        variance: Whether the input covariance is a diagonal matrix.
        stability: For accurate results this should be zero
            if used in training, use a value like 1e-4 for stability.

    Returns:
        Output variance of Affine-ReLU-Affine for Gaussian input
        with mean = `mean` and covariance matrix = `covariance`
        where the bias of the first affine = -A*`mean`.
    '''
    a_covariance = affine.covariance(covariance, A, variance=variance)
    r_covariance = relu.zero_mean_covariance(a_covariance, stability=stability)
    return affine.variance(r_covariance, B)


def special_covariance(covariance, A, B, variance=False, stability=0.0):
    '''Output covariance of Affine-ReLU-Affine for special Gaussian input.

    f(x) = B*max(A*x+c1, 0)+c2, where c1 = -A*input_mean.

    For this specific c1, this function doesn't depend on
    neither the input mean nor the biases.

    Args:
        covariance: Input covariance matrix (Batch, Size, Size)
            or variance vector (Batch, Size) for diagonal covariance.
        A: The A matrix (M, Size).
        B: The B matrix (N, M).
        variance: Whether the input covariance is a diagonal matrix.
        stability: For accurate results this should be zero
            if used in training, use a value like 1e-4 for stability.

    Returns:
        Output covariance of Affine-ReLU-Affine for Gaussian input
        with mean = `mean` and covariance matrix = `covariance`
        where the bias of the first affine = -A*`mean`.
    '''
    a_covariance = affine.covariance(covariance, A, variance=variance)
    r_covariance = relu.zero_mean_covariance(a_covariance, stability=stability)
    return affine.covariance(r_covariance, B)
