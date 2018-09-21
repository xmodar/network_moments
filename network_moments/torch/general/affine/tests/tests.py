import torch
from unittest import TestCase
from ....utils import rand, cov
from ..affine import (mean, variance, covariance)


__all__ = ['Tightness', 'Batch']


class Tightness(TestCase):
    '''Test the tightness against Monte-Carlo estimations.

    The expressions are for the affine transformation f(x) = A * x + b.
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

        # variables
        self.A = torch.randn(length, length, dtype=dtype, device=device)
        self.b = torch.randn(length, dtype=dtype, device=device)

        # input mean and covariance
        self.mu = torch.randn(length, dtype=dtype, device=device) * factor
        self.cov = rand.definite(length, dtype=dtype, device=device,
                                 positive=True, semi=False, norm=factor**2)

        # Monte-Carlo estimation of the output mean and variance
        normal = torch.distributions.MultivariateNormal(self.mu, self.cov)
        samples = normal.sample((count,))
        out_samples = samples.matmul(self.A.t()) + self.b
        self.mc_mu = torch.mean(out_samples, dim=0)
        self.mc_var = torch.var(out_samples, dim=0)
        self.mc_cov = cov(out_samples)

    def tearDown(self):
        del (self.A, self.b, self.mu, self.cov,
             self.mc_cov, self.mc_var, self.mc_mu)

    def test_mean(self):
        mu = mean(self.mu, self.A, self.b)
        self.assertTrue(torch.allclose(mu, self.mc_mu, rtol=1e-1))
        return self.mc_mu, mu

    def test_variance(self):
        var = variance(self.cov, self.A)
        self.assertTrue(torch.allclose(var, self.mc_var, rtol=1e-1))
        return self.mc_var, var

    def test_covariance(self):
        cov = covariance(self.cov, self.A)
        self.assertTrue(torch.allclose(cov, self.mc_cov, rtol=1e-1))
        return self.mc_cov, cov


class Batch(TestCase):
    '''Test the correctness of batch implementation.'''

    def setUp(self, size=(2, 5), batch=3, dtype=torch.float64, device=None,
              seed=None, mu=None, cov=None, A=None, b=None):
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
            A: To test a specific A matrix.
            b: To test a specific bias b.
        '''
        if seed is not None:
            torch.manual_seed(seed)
        if A is None:
            A = torch.rand(size, dtype=dtype, device=device)
        if b is None:
            b = torch.rand(size[0], dtype=dtype, device=device)
        if mu is None:
            mu = torch.rand(size[1], dtype=dtype, device=device)
        if cov is None:
            cov = rand.definite(size[1], dtype=dtype, device=device,
                                positive=True, semi=False, norm=10**2)
        self.A = A
        self.b = b
        var = torch.diag(cov)
        self.batch_mean = torch.stack([(i + 1) * mu for i in range(batch)])
        self.batch_cov = torch.stack([(i + 1) * cov for i in range(batch)])
        self.batch_var = torch.stack([(i + 1) * var for i in range(batch)])

    def tearDown(self):
        del (self.A, self.b, self.batch_mean, self.batch_var, self.batch_cov)

    def test_mean(self):
        single_mean = [mean(c, self.A, self.b) for c in self.batch_mean]
        batch_mean = mean(self.batch_mean, self.A, self.b)
        close = [torch.allclose(r, c) for r, c in zip(batch_mean, single_mean)]
        self.assertNotIn(False, close)
        return single_mean, batch_mean

    def test_covariance(self):
        single_cov = [covariance(c, self.A) for c in self.batch_cov]
        batch_cov = covariance(self.batch_cov, self.A)
        close = [torch.allclose(r, c) for r, c in zip(batch_cov, single_cov)]
        self.assertNotIn(False, close)
        return single_cov, batch_cov

    def test_diagonal_covariance(self):
        single_cov = [covariance(c, self.A, True) for c in self.batch_var]
        batch_cov = covariance(self.batch_var, self.A, True)
        close = [torch.allclose(r, c)
                 for r, c in zip(batch_cov, single_cov)]
        self.assertNotIn(False, close)
        return single_cov, batch_cov

    def test_variance(self):
        single_cov = [variance(c, self.A) for c in self.batch_cov]
        batch_cov = variance(self.batch_cov, self.A)
        close = [torch.allclose(r, c) for r, c in zip(batch_cov, single_cov)]
        self.assertNotIn(False, close)
        return single_cov, batch_cov

    def test_diagonal_variance(self):
        single_cov = [variance(c, self.A, True) for c in self.batch_var]
        batch_cov = variance(self.batch_var, self.A, True)
        close = [torch.allclose(r, c) for r, c in zip(batch_cov, single_cov)]
        self.assertNotIn(False, close)
        return single_cov, batch_cov
