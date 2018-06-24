import tensorflow as tf
from ...utils import matmul, mul_diag


def mean(mean, A, b=None, transposed=False):
    '''Output mean of Affine for general input.

    f(x) = A * x + b.

    Args:
        mean: Input mean of size (Batch, Size).
        A: Matrix of size (M, Size).
        b: Bias of size (Batch, M).
        transposed: Whether the A matrix is transposed (Size, M).

    Returns:
        Output mean of Affine for general input (Batch, M).
    '''
    with tf.name_scope('affine_mean'):
        Am = matmul(mean, A, transpose_b=not transposed)
        return Am if b is None else Am + b


def covariance(covariance, A, variance=False, At=None):
    '''Output covariance matrix of Affine for general input.

    f(x) = A * x + b.

    Args:
        covariance: Input covariance matrix of size (Batch, Size, Size)
            or variance of size (Batch, Size).
        A: Matrix (M, Size).
        variance: Whether the input covariance is a diagonal matrix.
        At: The precomputed tf.transpose(A).

    Returns:
        Output covariance of Affine for general input (Batch, M, M).
    '''
    with tf.name_scope('affine_covariance'):
        if variance:
            AS = mul_diag(A, covariance)
        else:
            AS = matmul(A, covariance)
        return matmul(AS, tf.transpose(A) if At is None else At)


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
    with tf.name_scope('affine_variance'):
        if variance:
            AS = mul_diag(A, covariance)
        else:
            AS = matmul(A, covariance)
        return tf.reduce_sum(AS * A, -1)
