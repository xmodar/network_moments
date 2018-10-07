import torch
from ...utils import map_batch, sqrtm, mul_diag, linearize, jac_at_x


__all__ = ['mean', 'covariance', 'variance', 'batch_moments']


def batch_moments(f, mu, var, jacobian=False,
                  mean=True, covariance=True, diagonal=False):
    '''Compute the mean and covariance of Affine applied on Gaussian input.

    Args:
        f: Affine function or a function to be linearized around `mu`.
        mu: Input mean (Batch, *size).
        var: Input covariance matrix or variance (Batch, size[, size]).
        jacobian: Whether to internally compute the Jacobian of `f`.
            Setting this to False is much more efficient if size is big.
        mean: Whether to output the mean.
        covariance: Whether to output the covariance.
        diagonal: Whether to output the variance if `covariance` is True.

    Returns:
        Output mean and covariance of Affine applied on Gaussian input.
    '''
    if covariance:
        batch = mu.size(0)
        if batch != var.size(0) and 1 not in (batch, var.size(0)):
            msg = 'Both input mean and variance must have a batch dim'
            raise ValueError(msg)

        # try to obtain A, the Jacobian of f at mu
        A = None
        if isinstance(f, torch.nn.Linear):
            A = f.weight.unsqueeze(0)
        elif jacobian:
            A = linearize(f, mu, jacobian_only=True)

        # otherwise, compute S = sqrt(var)
        elif var.dim() <= 2:
            var = map_batch(lambda x: x.sqrt().diag(), var)
        else:
            var = map_batch(sqrtm, var)

        if A is not None:
            # we have the Jacobian matrix of f
            v = variance if diagonal else _covariance
            a_var = var.dim() <= 2
            var = map_batch(lambda c, a: v(c, a, variance=a_var), var, A)
        else:
            # we will compute AS = (A * var) using jac_at_x()
            m = mu.unsqueeze(1)
            var = var.view(-1, var.size(1), *mu.size()[1:])
            AS = map_batch(jac_at_x, [f], m, var)
            AS = AS.view(*AS.size()[:2], -1)
            var = ((AS**2).sum(1) if diagonal
                   else AS.transpose(-1, -2).bmm(AS))
        if not mean:
            return var
    if mean:
        mu = f(mu)
        if not covariance:
            return mu
    return mu, var


def mean(mean, A, b=0.0):
    '''Output mean of Affine for general input.

    f(x) = A * x + b.

    Args:
        mean: Input mean of size (Batch, Size).
        A: Matrix of size (M, Size).
        b: Bias of size (Batch, M).

    Returns:
        Output mean of Affine for general input (Batch, M).
    '''
    Am = mean.matmul(A.t())
    return Am if b is None else Am + b


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
    if variance:
        if covariance.dim() == 1:
            return (A * A).mv(covariance)
        else:
            return covariance.matmul((A * A).t())
    else:
        return torch.sum(A.matmul(covariance) * A, -1)


_covariance = covariance  # to use covariance inside batch_moments()
