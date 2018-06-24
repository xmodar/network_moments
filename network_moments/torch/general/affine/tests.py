import torch
from ...utils import rand
from .affine import (mean, variance, covariance)


def tightness(length=1, count=1000000, seed=None,
              dtype=torch.float64, device=None):
    '''Test the tightness of the expressions against Monte-Carlo estimations.

    The expressions are for the affine transformation f(x) = A * x + b.

    Args:
        length: Size of the vector.
        count: Number of samples for Monte-Carlo estimation.
        seed: Seed for the random number generator.
        dtype: The data type.
        device: In which device.

    Returns:
        (out_mu, mc_mu), (out_var, mc_var)
        out_mu: The output mean computed using the expressions.
        mc_mu: The output mean estimated using Monte-Carlo sampling.
        out_var: The output variance computed using the expressions.
        mc_var: The output variance estimated using Monte-Carlo sampling.
    '''
    if seed is not None:
        torch.manual_seed(seed)

    # variables
    A = torch.randn(length, length, dtype=dtype, device=device)
    b = torch.randn(length, dtype=dtype, device=device)

    # input mean and covariance
    mu = torch.randn(length, dtype=dtype, device=device)
    cov = rand.definite(length, dtype=dtype, device=device,
                        positive=True, semi=False, norm=1.0)

    # analytical output mean and variance
    out_mu = mean(mu, A, b)
    out_var = variance(cov, A)

    # Monte-Carlo estimation of the output mean and variance
    normal = torch.distributions.MultivariateNormal(mu, cov)
    samples = normal.sample((count,))
    out_samples = samples.matmul(A.t()) + b
    mc_mu = torch.mean(out_samples, dim=0)
    mc_var = torch.var(out_samples, dim=0)
    return (out_mu, mc_mu), (out_var, mc_var)


def batch_mean(size=(2, 5), batch=3, dtype=torch.float64, device=None,
               mu=None, A=None, b=None):
    '''Test the correctness of batch implementation of mean().

    This function will stack `[1 * mu, 2 * mu, ..., batch * mu]`.
    Then, it will see whether the batch output is accurate or not.

    Args:
        size: Tuple size of matrix A.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        mu: To test a specific mean mu.
        A: To test a specific A matrix.
        b: To test a specific bias b.

    Returns:
        A scalar, the closer it is to zero,
        the more accurate the implementation.
    '''
    if A is None:
        A = torch.rand(size, dtype=dtype, device=device)
    if b is None:
        b = torch.rand(A.size(0), dtype=dtype, device=device)
    if mu is None:
        mu = torch.rand(A.size(1), dtype=dtype, device=device)
    means = torch.stack([(i + 1) * mu for i in range(batch)])
    return max([torch.mean(torch.abs(r - mean(c, A, b)))
                for r, c in zip(mean(means, A, b), means)]).item()


def batch_covariance(size=(2, 5), batch=3, dtype=torch.float64, device=None,
                     cov=None, A=None):
    '''Test the correctness of batch implementation of covariance().

    This function will stack `[1 * cov, 2 * cov, ..., batch * cov]`.
    Then, it will see whether the batch output is accurate or not.

    Args:
        size: Tuple size of matrix A.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        cov: To test a specific covariance matrix.
        A: To test a specific A matrix.

    Returns:
        A tuple of two scalars(r1, r2), the closer they are to zero,
        the more accurate the implementation.
        r1: For full input covariance matrix.
        r2: For diagonal input covariance matrix.
    '''
    if A is None:
        A = torch.rand(size, dtype=dtype, device=None)
    if cov is None:
        cov = torch.rand(A.size(1), A.size(1), dtype=dtype, device=None)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    r1 = max([torch.mean(torch.abs(r - covariance(c, A)))
              for r, c in zip(covariance(covs, A), covs)]).item()
    cov = torch.diag(cov)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    r2 = max([torch.mean(torch.abs(r - covariance(c, A, True)))
              for r, c in zip(covariance(covs, A, True), covs)]).item()
    return r1, r2


def batch_variance(size=(2, 5), batch=3, dtype=torch.float64, device=None,
                   cov=None, A=None):
    '''Test the correctness of batch implementation of variance().

    This function will stack `[1 * cov, 2 * cov, ..., batch * cov]`.
    Then, it will see whether the batch output is accurate or not.

    Args:
        size: Tuple size of matrix A.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        cov: To test a specific covariance matrix.
        A: To test a specific A matrix.

    Returns:
        A tuple of two scalars(r1, r2), the closer they are to zero,
        the more accurate the implementation.
        r1: For full input covariance matrix.
        r2: For diagonal input covariance matrix.
    '''
    if A is None:
        A = torch.rand(size, dtype=dtype, device=device)
    if cov is None:
        cov = torch.rand(A.size(1), A.size(1), dtype=dtype, device=device)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    r1 = max([torch.mean(torch.abs(r - variance(c, A)))
              for r, c in zip(variance(covs, A), covs)]).item()
    cov = torch.diag(cov)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    r2 = max([torch.mean(torch.abs(r - variance(c, A, True)))
              for r, c in zip(variance(covs, A, True), covs)]).item()
    return r1, r2
