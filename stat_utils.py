import torch

__all__ = ['cov_mean', 'gaussian', 'rand_matrix', 'VarianceMeter']


def cov_mean(samples, unbiased=False, keepdim=False):
    """Compute the covariance matrix and mean of data samples."""
    mean = samples.mean(0, keepdim=True)
    factor = 1 / (samples.shape[0] - bool(unbiased))
    samples = samples - mean
    covariance = factor * (samples.transpose(-1, -2) @ samples)
    if keepdim:
        return covariance.unsqueeze(0), mean
    return covariance, mean.squeeze(0)


def rand_matrix(*shape, norm=None, trace=None, dtype=None, device=None):
    """Generate a random positive definite matrix."""
    assert None in (norm, trace), 'provide only a norm or a trace or neither'
    eigen = 1 - torch.rand(*shape, dtype=dtype, device=device)
    q = eigen.new(*eigen.shape, eigen.shape[-1]).normal_().qr().Q
    if norm is not None:
        eigen *= norm / eigen.norm(dim=-1, keepdim=True)
    elif trace is not None:
        eigen *= trace / eigen.sum(dim=-1, keepdim=True)
    return (q * eigen.unsqueeze(-2)) @ q.transpose(-1, -2)


def gaussian(covariance, mean=None):
    """Wrap torch.distributions.MultivariateNormal to any sample shape."""
    if not torch.is_tensor(covariance):
        covariance = rand_matrix(covariance)
    if mean is None:
        mean = covariance.new_zeros(covariance.shape[-1])
    dist = torch.distributions.MultivariateNormal(mean.view(-1), covariance)
    dist.draw = lambda *batch: dist.sample(batch).view(*batch, *mean.shape)
    dist.center = mean
    return dist


class VarianceMeter:
    """Estimate the variance for a stream of values."""

    def __init__(self, unbiased=True):
        """Initialize the meter."""
        self.unbiased = bool(unbiased)
        self.count = self.mean = self.mass = 0

    def reset(self):
        """Reset the meter."""
        self.__init__(self.unbiased)

    @property
    def factor(self):
        """Get inverted count or Bessel's correction factor."""
        count = self.count - int(bool(self.unbiased))
        return float('nan') if count == 0 else 1 / count

    @property
    def variance(self):
        """Get the variance."""
        return self.mass * self.factor

    @property
    def std(self):
        """Get the standard deviation."""
        return self.variance**0.5

    def update(self, mean, variance=None, count=1, unbiased=None):
        """Update the meter with batch statistics.

        Args:
            mean: Batch mean.
            variance: Batch variance.
            count: Batch size.
            unbiased: If `variance` is unbiased (default: `self.unbiased`).

        """
        assert isinstance(count, int) and count > 0
        diff = mean - self.mean
        self.mean += diff * (count / (self.count + count))
        diff *= mean - self.mean
        if count > 1:
            diff += variance
            diff *= count
            if unbiased is None:
                unbiased = self.unbiased
            if unbiased:
                diff -= variance
        self.mass += diff
        self.count += count
