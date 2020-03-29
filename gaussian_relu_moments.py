from math import pi, sqrt

from erf_exp_integral import erf_exp_integral

__all__ = [
    # main functions
    'relu_cov_mean',  # not differentiable for non-zero mean (for now)
    'relu_var_mean',
    'relu_mean',

    # auxiliary functions
    'relu_std_mean',
    'relu_cov',
    'relu_var',
    'relu_std',
    'forward_gaussian',
]


def relu_cov_mean(covariance, mean=None, num_terms=0):
    """Compute the covariance and mean of ReLU(gaussian_vector)."""
    d = lambda x: x.diagonal(0, -1, -2)  # diagonal of a matrix
    t = lambda x: x.transpose(-1, -2)  # transpose of a matrix
    o = lambda x, y: x.unsqueeze(-1) * y.unsqueeze(-2)  # vector outer product
    m = lambda x, y: x * y.unsqueeze(-2)  # multiply matrix columns by a vector

    std = d(covariance).sqrt()
    std_outer = o(std, std)
    if mean is None:
        mean = relu_mean(std)
        correlation = (covariance / std_outer).clamp(-1, 1)
        c = lambda x: (x * (-x).acos() + (1 - x.pow(2)).sqrt())
        o_correlation = (1 / (2 * pi)) * c(correlation) * std_outer
        return o_correlation - o(mean, mean), mean

    sigma_det = (std_outer - covariance).clamp(0) * (std_outer + covariance)

    mean_outer, std_x_mean = o(mean, mean), o(std, mean)
    q = lambda x, y, z: (x + t(x) - 2 * y) / z
    quadratic = q(std_x_mean.pow(2), mean_outer * covariance, sigma_det)

    sqrt_sigma_det = sigma_det.sqrt()
    term1 = (1 / (2 * pi)) * sqrt_sigma_det * (-0.5 * quadratic).exp()

    mean_std = sqrt(1 / 2) * (mean / std)
    n = lambda x, y, z: (sqrt(1 / 2) * t(x) - y) / z
    sigma_tilde_mean = n(std_x_mean, m(covariance, mean_std), sqrt_sigma_det)

    pdf = (1 / sqrt(2 * pi)) * (-mean_std.pow(2)).exp()
    term2 = (1 / 2) * m(t(std_x_mean), pdf) * (1 + sigma_tilde_mean.erf())

    cdf = (1 / 2) * (1 + mean_std.erf())
    mean_sigma = mean_outer + covariance
    term4 = (1 / 2) * m(mean_sigma, cdf)

    omega = term1 + term2 + t(term2) + term4

    irho = std_outer / sqrt_sigma_det
    a = irho * covariance / std_outer
    b = irho * mean_std.unsqueeze(-1)
    c = -mean_std.unsqueeze(-2)
    b[(irho == float('inf')).expand_as(b)] = 0
    integration = erf_exp_integral(a, b, c, num_terms - 1)

    correlation = omega + (1 / sqrt(4 * pi)) * mean_sigma * integration
    second_moment = d(mean_sigma) * cdf + d(std_x_mean) * pdf
    d(correlation).copy_(second_moment)

    o_mean = mean * cdf + std * pdf
    o_covariance = correlation - o(o_mean, o_mean)
    return o_covariance, o_mean


def relu_var_mean(std, mean=None):
    """Compute the variance and mean of ReLU(gaussian_vector)."""
    if mean is None:
        o_mean = relu_mean(std)
        return o_mean * std * ((pi - 1) / sqrt(2 * pi)), o_mean
    d_std_x_mean = std * mean
    d_mean_sigma = mean.pow(2) + std.pow(2)
    mean_std = sqrt(1 / 2) * mean / std
    pdf = (1 / sqrt(2 * pi)) * (-mean_std.pow(2)).exp()
    cdf = (1 / 2) * (1 + mean_std.erf())
    o_mean = mean * cdf + std * pdf
    second_moment = d_mean_sigma * cdf + d_std_x_mean * pdf
    return second_moment - o_mean.pow(2), o_mean


def relu_mean(std, mean=None):
    """Compute the mean of ReLU(gaussian_vector)."""
    if mean is None:
        return std * (1 / sqrt(2 * pi))
    mean_std = sqrt(1 / 2) * mean / std
    pdf = (1 / sqrt(2 * pi)) * (-mean_std.pow(2)).exp()
    cdf = (1 / 2) * (1 + mean_std.erf())
    return mean * cdf + std * pdf


def relu_std_mean(std, mean=None):
    """Compute the standard deviation and mean of ReLU(gaussian_vector)."""
    variance, mean = relu_var_mean(std, mean)
    return variance.sqrt(), mean


def relu_cov(covariance, mean=None, num_terms=0):
    """Compute the covariance of ReLU(gaussian_vector)."""
    return relu_cov_mean(covariance, mean, num_terms)[0]


def relu_var(std, mean=None):
    """Compute the variance of ReLU(gaussian_vector)."""
    return relu_var_mean(std, mean)[0]


def relu_std(std, mean=None):
    """Compute the standard deviation of ReLU(gaussian_vector)."""
    return relu_var_mean(std, mean)[0].sqrt()


def forward_gaussian(model, covariance, mean, terms=5):
    """Compute variance and mean of Affine-ReLU-Affine for gaussian input."""
    a, _, b = model  # assumes Sequential(Linear, ReLU, Linear)
    covariance, mean = a.weight @ covariance @ a.weight.t(), a(mean)
    if terms >= 0:
        covariance, mean = relu_cov_mean(covariance, mean, terms)
    else:  # assume zero mean relu input to compute off-diagonals
        std = covariance.diagonal(0, -1, -2).sqrt()
        variance, mean = relu_var_mean(std, mean)
        covariance = relu_cov(covariance)
        covariance.diagonal(0, -1, -2).copy_(variance)
    variance, mean = ((b.weight @ covariance) * b.weight).sum(-1), b(mean)
    return variance, mean
