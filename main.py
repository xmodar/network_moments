import torch

from gaussian_relu_moments import (rand_matrix, relu_cov_mean, relu_mean,
                                   relu_var_mean)


def cov_mean(dim, cov_batch=0, mean_batch=0, zero_mean=False):
    print(dict(cbatch=cov_batch, mbatch=mean_batch, zmean=zero_mean))
    cbatch = () if cov_batch == 0 else (cov_batch,)
    mbatch = () if mean_batch == 0 else (mean_batch,)
    covariance = rand_matrix(*cbatch, dim, dtype=torch.float64)
    mean = covariance.new(*mbatch, dim) * (not zero_mean)
    return covariance, mean


def test_with_cov_mean(dim, cov_batch=0, mean_batch=0, zero_mean=False):
    covariance, mean = cov_mean(dim, cov_batch, mean_batch, zero_mean)
    std = covariance.diagonal(0, -1, -2).sqrt()

    r_covariance, r_mean = relu_cov_mean(covariance, mean)
    r_var = r_covariance.diagonal(0, -1, -2)

    o_mean = relu_mean(std, None if zero_mean else mean)
    assert torch.allclose(o_mean, r_mean), 'mean'

    o_var, o_mean = relu_var_mean(std, None if zero_mean else mean)
    assert torch.allclose(o_var, r_var), 'var'
    assert torch.allclose(o_mean, r_mean), 'var_mean'

    if zero_mean:
        o_covariance, o_mean = relu_cov_mean(covariance)
        assert torch.allclose(o_covariance, r_covariance), 'cov'
        assert torch.allclose(o_mean, r_mean), 'cov_mean'
    else:
        test_with_cov_mean(dim, cov_batch, mean_batch, zero_mean=True)


def test_cov_with_terms(dim, cov_batch=0, mean_batch=0, zero_mean=False):
    covariance, mean = cov_mean(dim, cov_batch, mean_batch, zero_mean)
    r_covariance, _ = relu_cov_mean(covariance, mean)

    def error(a, b, rtol=1e-5, atol=1e-8):
        return ((a - b).abs() - (atol + rtol * b.abs())).clamp_(0)

    for num_terms in range(1, 51):
        o_covariance, _ = relu_cov_mean(covariance, mean, num_terms)
        value = error(o_covariance, r_covariance).mean().item()
        print(num_terms, value)
        if value == 0:
            break


def test_batching(function, dim):
    function(dim, cov_batch=0, mean_batch=0)
    function(dim, cov_batch=1, mean_batch=1)
    function(dim, cov_batch=3, mean_batch=3)
    function(dim, cov_batch=1, mean_batch=3)
    function(dim, cov_batch=3, mean_batch=1)
    function(dim, cov_batch=0, mean_batch=1)
    function(dim, cov_batch=1, mean_batch=0)
    function(dim, cov_batch=0, mean_batch=3)
    function(dim, cov_batch=3, mean_batch=0)


if __name__ == '__main__':
    test_batching(test_with_cov_mean, dim=5)
    test_batching(test_cov_with_terms, dim=5)
