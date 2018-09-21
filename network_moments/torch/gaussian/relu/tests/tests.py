import torch
from unittest import TestCase
from ....utils import rand, cov, outer
from ..relu import (mean, variance, zero_mean_covariance)


__all__ = ['Tightness', 'Batch']


class Tightness(TestCase):
    '''Test the tightness against Monte-Carlo estimations.

    The expressions are for the ReLU function f(x) = max(x, 0).
    '''

    def setUp(self, length=3, factor=10, count=1000000,
              seed=None, dtype=torch.float64, device=None):
        '''Set up the test values.

        Args:
            length: Size of the vector.
            factor: To multiply the mean and standard deviation.
            count: Number of samples for Monte-Carlo estimation.
            seed: Seed for the random number generator.
            dtype: The data type.
            device: In which device.
        '''
        if seed is not None:
            torch.manual_seed(seed)

        # input mean and covariance
        self.mu = torch.randn(length, dtype=dtype, device=device) * factor
        self.cov = rand.definite(length, dtype=dtype, device=device,
                                 positive=True, semi=False, norm=factor**2)
        self.var = self.cov.diag()

        # Monte-Carlo estimation of the output mean and variance
        normal = torch.distributions.MultivariateNormal(self.mu, self.cov)
        out_samples = normal.sample((count,)).clamp_(min=0.0)
        self.mc_mu = torch.mean(out_samples, dim=0)
        self.mc_var = torch.var(out_samples, dim=0)
        normal = torch.distributions.MultivariateNormal(self.mu * 0, self.cov)
        out_samples = normal.sample((count,)).clamp_(min=0.0)
        mean = torch.mean(out_samples, dim=0)
        self.mc_zm_cov = cov(out_samples)
        self.mc_zm_corr = self.mc_zm_cov + outer(mean)

    def tearDown(self):
        del (self.mu, self.cov, self.var,
             self.mc_mu, self.mc_var,
             self.mc_zm_cov, self.mc_zm_corr)

    def test_mean(self):
        mu = mean(self.mu, self.var)
        self.assertTrue(torch.allclose(mu, self.mc_mu, rtol=1e-1))
        return self.mc_mu, mu

    def test_variance(self):
        var = variance(self.mu, self.var)
        self.assertTrue(torch.allclose(var, self.mc_var, rtol=1e-1))
        return self.mc_var, var

    def test_zero_mean_covariance(self):
        cov = zero_mean_covariance(self.cov)
        self.assertTrue(torch.allclose(cov, self.mc_zm_cov, rtol=1e-1))
        return self.mc_zm_cov, cov


class Batch(TestCase):
    '''Test the correctness of batch implementation.'''

    def setUp(self, size=(2, 5), batch=3, dtype=torch.float64, device=None,
              seed=None, mu=None, cov=None):
        '''Test the correctness of batch implementation of mean().

        This function will stack `[1 * mu, 2 * mu, ..., batch * mu]`.
        Then, it will see whether the batch output is accurate or not.

        Args:
            size: Tuple size of matrix A.
            batch: The batch size > 0.
            dtype: data type.
            device: In which device.
            seed: Seed for the random number generator.
            mu: To test a specific mean mu.
            cov: To test a specific covariance matrix.
        '''
        if seed is not None:
            torch.manual_seed(seed)
        if mu is None:
            mu = torch.rand(size[1], dtype=dtype, device=device)
        if cov is None:
            cov = rand.definite(size[1], dtype=dtype, device=device,
                                positive=True, semi=False, norm=10**2)
        var = torch.diag(cov)
        self.mu = mu
        self.cov = cov
        self.var = var
        self.batch_mean = torch.stack([(i + 1) * mu for i in range(batch)])
        self.batch_var = torch.stack([(i + 1) * var for i in range(batch)])
        self.batch_cov = torch.stack([(i + 1) * cov for i in range(batch)])

    def tearDown(self):
        del (self.mu, self.var, self.cov,
             self.batch_mean, self.batch_var, self.batch_cov)

    def test_mean_batch_mean(self):
        single_mean = [mean(c, self.var) for c in self.batch_mean]
        batch_mean = mean(self.batch_mean, self.var)
        close = [torch.allclose(r, c) for r, c in zip(batch_mean, single_mean)]
        self.assertNotIn(False, close)
        return single_mean, batch_mean

    def test_mean_batch_variance(self):
        single_mean = [mean(self.mu, c) for c in self.batch_var]
        batch_mean = mean(self.mu, self.batch_var)
        close = [torch.allclose(r, c) for r, c in zip(batch_mean, single_mean)]
        self.assertNotIn(False, close)
        return single_mean, batch_mean

    def test_mean_batch_mean_and_variance(self):
        single_mean = [mean(c, v)
                       for c, v in zip(self.batch_mean, self.batch_var)]
        batch_mean = mean(self.batch_mean, self.batch_var)
        close = [torch.allclose(r, c) for r, c in zip(batch_mean, single_mean)]
        self.assertNotIn(False, close)
        return single_mean, batch_mean

    def test_variance_batch_mean(self):
        single_var = [variance(c, self.var) for c in self.batch_mean]
        batch_var = variance(self.batch_mean, self.var)
        close = [torch.allclose(r, c) for r, c in zip(batch_var, single_var)]
        self.assertNotIn(False, close)
        return single_var, batch_var

    def test_variance_batch_variance(self):
        single_var = [variance(self.mu, c) for c in self.batch_var]
        batch_var = variance(self.mu, self.batch_var)
        close = [torch.allclose(r, c) for r, c in zip(batch_var, single_var)]
        self.assertNotIn(False, close)
        return single_var, batch_var

    def test_variance_batch_mean_and_variance(self):
        single_var = [variance(c, v)
                      for c, v in zip(self.batch_mean, self.batch_var)]
        batch_var = variance(self.batch_mean, self.batch_var)
        close = [torch.allclose(r, c) for r, c in zip(batch_var, single_var)]
        self.assertNotIn(False, close)
        return single_var, batch_var

    def test_zero_mean_covariance(self):
        single_cov = [zero_mean_covariance(c) for c in self.batch_cov]
        batch_cov = zero_mean_covariance(self.batch_cov)
        close = [torch.allclose(r, c) for r, c in zip(batch_cov, single_cov)]
        self.assertNotIn(False, close)
        return single_cov, batch_cov
