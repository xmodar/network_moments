import torch
from ..ops import (mul_diag, normalize_)


__all__ = ['from_eigen', 'definite']


def from_eigen(eigen):
    '''Construct a random matrix with given the eigenvalues.

    To construct such a matrix form the eigenvalue decomposition,
    (i.e. U * Sigma * U.t()), we need to find a unitary matrix U
    and Sigma is the diagonal matrix of the eigenvalues `eigen`.
    The matrix U can be the unitary matrix Q from
    the QR-decomposition of a randomly generated matrix.

    Args:
        eigen: A vector of size (Batch, Size).

    Returns:
        A random matrix of size (Batch, Size, Size).
    '''
    size = eigen.size(-1)
    Q, _ = torch.qr(torch.randn(
        (size, size), dtype=eigen.dtype, device=eigen.device))
    return mul_diag(Q, eigen).matmul(Q.t())


def definite(size, batch=None, norm=None,
             positive=True, semi=False, dtype=None, device=None):
    '''Random definite matrix.

    A positive/negative definite matrix is a matrix
    with positive/negative eigenvalues, respectively.
    They are called semi-definite if the eigenvalues are allowed to be zeros.
    The eigenvalues are some random vector of unit norm.
    This vector is what control (positive vs. negative)
    and (semi-definite vs definite).
    We multiply this vector by the desired `norm`.

    Args:
        size: The output matrix is of size (`size`, `size`).
        batch: Number of matrices to generate.
        norm: The Frobenius norm of the output matrix.
        positive: Whether positive-definite or negative-definite.
        semi: Whether to construct semi-definite or definite matrix.
        dtype: The data type.
        device: In which device.

    Returns:
        Random definite matrices of size (`Batch`, `size`, `size`)
        and Frobenius norm `norm`.
    '''
    shape = size if batch is None else (batch, size)
    eigen = torch.rand(shape, dtype=dtype, device=device)
    if not semi:
        eigen = 1.0 - eigen
    if not positive:
        eigen = -eigen
    eigen = eigen if norm is None else normalize_(eigen, norm)
    return from_eigen(eigen)
