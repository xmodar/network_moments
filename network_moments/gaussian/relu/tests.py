import torch
from ...utils import (rand, diagonal)
from .relu import (mean, zero_mean_correlation, zero_mean_covariance)


def tightness(length=1, count=1000000, seed=None,
              dtype=torch.float64, device=None):
    '''Test the tightness of the expressions against Monte-Carlo estimations.

    The expressions are for the ReLU function f(x) = max(x, 0).

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

    # input mean and covariance
    mu = torch.zeros(length, dtype=dtype, device=device)
    cov = rand.definite(length, dtype=dtype, device=device,
                        positive=True, semi=False, norm=1.0)

    # analytical output mean and variance
    out_mu = mean(mu, diagonal(cov, dim1=-2, dim2=-1))
    out_var = diagonal(zero_mean_covariance(cov), dim1=-2, dim2=-1)

    # Monte-Carlo estimation of the output mean and variance
    normal = torch.distributions.MultivariateNormal(mu, cov)
    samples = normal.sample((count,))
    out_samples = torch.max(samples,
                            torch.zeros([], dtype=dtype, device=device))
    mc_mu = torch.mean(out_samples, dim=0)
    mc_var = torch.var(out_samples, dim=0)
    return (out_mu, mc_mu), (out_var, mc_var)


def batch_mean(size=5, batch=3, dtype=torch.float64, device=None,
               mu=None, var=None):
    '''Test the correctness of batch implementation of mean().

    This function will stack `[1 * mu, 2 * mu, ..., batch * mu]`.
    Then, it will see whether the batch output is accurate or not.
    It will do the same for the variance as well.

    Args:
        size: Scalar size of the input mean.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        mu: To test a specific mean mu.
        var: To test a specific variance var.

    Returns:
        A tuple of three scalars(r1, r2, r3), the closer they are to zero,
        the more accurate the implementation.
        r1: For batch mean.
        r2: For batch variance.
        r3: For batch mean and batch variance.
    '''
    if mu is None:
        mu = torch.rand(size, dtype=dtype, device=device)
    if var is None:
        var = torch.rand(size, dtype=dtype, device=device)
    means = torch.stack([(i + 1) * mu for i in range(batch)])
    vars = torch.stack([(i + 1) * var for i in range(batch)])
    r1 = max([torch.mean(torch.abs(r - mean(c, var)))
              for r, c in zip(mean(means, var), means)]).item()
    r2 = max([torch.mean(torch.abs(r - mean(mu, c)))
              for r, c in zip(mean(mu, vars), vars)]).item()
    r3 = max([torch.mean(torch.abs(r - mean(c, v)))
              for r, c, v in zip(mean(means, vars), means, vars)]).item()
    return r1, r2, r3


def batch_zero_mean_correlation(size=5, batch=3,
                                dtype=torch.float64, device=None, cov=None):
    '''Test the correctness of batch implementation of zero_mean_correlation().

    This function will stack `[1 * cov, 2 * cov, ..., batch * cov]`.
    Then, it will see whether the batch output is accurate or not.

    Args:
        size: Scalar size of the input mean.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        cov: To test a specific correlation matrix.

    Returns:
        A scalar, the closer they are to zero,
        the more accurate the implementation.
    '''
    if cov is None:
        cov = rand.definite(size, dtype=dtype, device=device,
                            positive=True, semi=False, norm=1.0)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    return max([torch.mean(torch.abs(r - zero_mean_correlation(c)))
                for r, c in zip(zero_mean_correlation(covs), covs)]).item()


def batch_zero_mean_covariance(size=5, batch=3,
                               dtype=torch.float64, device=None, cov=None):
    '''Test the correctness of batch implementation of zero_mean_covariance().

    This function will stack `[1 * cov, 2 * cov, ..., batch * cov]`.
    Then, it will see whether the batch output is accurate or not.

    Args:
        size: Scalar size of the input mean.
        batch: The batch size > 0.
        dtype: data type.
        device: In which device.
        cov: To test a specific covariance matrix.

    Returns:
        A scalar, the closer they are to zero,
        the more accurate the implementation.
    '''
    if cov is None:
        cov = rand.definite(size, dtype=dtype, device=device,
                            positive=True, semi=False, norm=1.0)
    covs = torch.stack([(i + 1) * cov for i in range(batch)])
    return max([torch.mean(torch.abs(r - zero_mean_covariance(c)))
                for r, c in zip(zero_mean_covariance(covs), covs)]).item()
