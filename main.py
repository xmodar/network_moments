from math import gamma, pi, sqrt

import torch
from scipy.special import gammainc as scipy_gammainc
from scipy.special import hermite as scipy_hermite
from torch.autograd import Function


def rand_cov(dim: int, definite=True, decomposed=False, **kwargs):
    """Generate a random covariance matrix."""
    if decomposed:
        scale_tril = torch.randn(dim, dim, **kwargs).tril()
        scale_tril.diagonal().abs_()
        cov = scale_tril @ scale_tril.transpose(-2, -1)
        std = cov.diagonal().sqrt().clamp_min(1e-12)
        cor = cov / (std.unsqueeze(-1) @ std.unsqueeze(-2))
        # scale_tril = torch.cholesky(cov)
        return cov, cor, std, scale_tril
    if definite:
        eigen = 1 - torch.rand(dim, **kwargs)
        eigen *= (2 * dim)**0.5 / eigen.norm()
        q, _ = torch.randn(dim, dim, **kwargs).qr()
        cov = (q * eigen.unsqueeze(0)) @ q.t()
    else:
        std = torch.randn(dim, **kwargs).abs() * (2 / dim)**0.25
        cov = std.unsqueeze(1) @ std.unsqueeze(0)
        cor = torch.rand_like(cov) * torch.randn_like(cov).sign()
        cor.triu_().diagonal().fill_(1)
        cor += cor.triu(1).t()
        cov *= cor
    return cov


def normpdf(x):
    return 1 / sqrt(2 * pi) * (x * x * -0.5).exp()


def normcdf(x):
    return 0.5 * (1 + (x * sqrt(0.5)).erf())


def polynomial(coefficients, x):
    """Evaluate a polynomial using Horner method.

    The coefficients are ordered from highest to lowest order.

    Args:
        coefficients: Tensor of size (N, *K).
            K is any broadcastable size to `x.size()`.
        x: Arbitrary tensor.

    Returns:
        The evaluated polynomial at `x` with the same size.

    """
    out = x.new_zeros(x.size()) if torch.is_tensor(x) else 0
    coefficients = iter(coefficients)
    out += next(coefficients, 0)
    for c in coefficients:
        out *= x
        out += c
    return out


def hermite(n, x):
    return polynomial(scipy_hermite(n).coeffs.tolist(), x)


def non_differentiable(function):
    """Decorate a function as non differentiable."""
    name = function.__qualname__

    @staticmethod
    def forward(ctx, *args, **kwargs):  # pylint: disable=unused-argument
        with torch.no_grad():
            return function(*args, **kwargs)

    return type(name, (Function,), {'forward': forward}).apply


@non_differentiable
def gammainc(a, x):
    return x.new(scipy_gammainc(a, x.cpu().numpy()))


class ExpErfIntegral:
    """The integral of `exp(-x**2) * erf(a*x+b)` from `c` to infinity."""

    @staticmethod
    def sum_until(iterable, end):
        """Sum the first few values of an iterable."""
        out = 0
        for _, e in zip(range(end), iterable):
            out += e
        return out

    @staticmethod
    def series(a, b, c, d=None, start=0):
        """Generate the infinite series in evaluate_integral().

        d = c if d is None else d.
        term(x, y, z) = x * a**y / gamma(z).

        f0(i) = gammainc(i + 3 / 2, c**2) * sign(d).
        f1(i) = 1 - gammainc(i + 1, c**2).
        s0(i) = term(hermite(2 * i + 1, b), 2 * i + 2, i + 2).
        s1(i) = term(hermite(2 * i, b), 2 * i + 1, i + 3 / 2).
        s(i) = f0(i) * s0(i) + f1(i) * s1(i).

        Since gamma and gammainc are related,
        we might be able to simplify this series:
        https://www.boost.org/doc/libs/1_71_0/libs/math/doc/html/math_toolkit/sf_gamma/igamma.html

        Yields:
            s(i) for every i from `start` to infinity.

        """
        i = start
        if d is None:
            d = c
        d = d.sign()
        c = c * c
        term = lambda x, y, z: x * a**y / gamma(z)
        while True:
            p_h = hermite(2 * i, b)
            c_h = hermite(2 * i + 1, b)
            s0 = term(c_h, 2 * i + 2, i + 2)
            s1 = term(p_h, 2 * i + 1, i + 3 / 2)
            f0 = d * gammainc(i + 3 / 2, c)
            f1 = 1 - gammainc(i + 1, c)
            yield f0 * s0 + f1 * s1
            i += 1

    @classmethod
    def evaluate(cls, a, b, c, n):
        """Integrate `exp(-x**2) * erf(a*x+b)` from `c` to infinity.

        The derivative of erf(x) is 2/sqrt(pi) * exp(-x**2).
        The derivative of erf(a*x+b) is 2a/sqrt(pi) * exp(-(a*x+b)**2).
        This function is very close of being a function multiplied by its derivative.
        We know that the integral of g(x) times its derivative is g(x)**2/2.
        This fact might be useful but we are not using it for the time being.

        """
        u = sqrt(pi) / 2
        absa = a.abs() <= 1
        v = a.where(absa, 1 / a)
        x = a.where(absa, v.abs()) * 0.5
        y = b.where(absa, -b * v)
        z = c.where(absa, a * c + b)
        w = c.where(absa, a.sign() * z)
        series = cls.sum_until(cls.series(x, y, z, w), end=n + 1)
        i0 = (2 * normcdf(sqrt(2) * z.abs()) - 1) * (0.5 -
                                                     normcdf(-sqrt(2) * y))
        term1 = (-y * y).exp() * series + (-sqrt(pi)) * w.sign() * i0
        term2 = (y / (1 + v * v).sqrt()).erf()
        terms = 1 - (term2 + term1 * (1 / u))
        iabsa = u * (a.sign() * terms - c.erf() * z.erf())
        nabsa = u * term2 + term1
        return torch.where(absa, iabsa, nabsa)


def cross_correlation(ux, uy, sx, sy, r, n):
    """Compute E[max(x,0)*max(y,0)] where x and y are jointly gaussian.

    The correlation is `r * sx * sy`.

    Args:
        ux: The mean of x.
        uy: The mean of y.
        sx: The standard deviation of x.
        sy: The standard deviation of y.
        r: The correlation coefficient.
        n: The number of terms used for the infinite series.

    Returns:
        E[max(x,0)*max(y,0)].
    """
    ir = (1 - r * r).sqrt()
    ir_ = 1 / ir
    sxy = sx * sy
    corr = sxy * r
    icorr = sxy * ir
    icorr_ = 1 / icorr

    uxy = ux * uy
    xy = ux * sy
    yx = uy * sx
    y_s = uy / sy
    x_s = ux / sx

    a = r * ir_
    b = (1 / sqrt(2)) * x_s * ir_
    c = (-1 / sqrt(2)) * y_s
    I = 1 / sqrt(4 * pi) * ExpErfIntegral.evaluate(a, b, c, n)
    t_1_ = (xy * xy + yx * yx + (-2) * uxy * corr).sqrt()
    t_1 = 1 / sqrt(2 * pi) * icorr * normpdf(t_1_ * icorr_)
    t_2 = (uxy + corr) * (normcdf(y_s) * 0.5 + I)
    t_3 = yx * normpdf(x_s) * normcdf((yx - r * xy) * icorr_)
    t_4 = xy * normpdf(y_s) * normcdf((xy - r * yx) * icorr_)
    return t_1 + t_2 + t_3 + t_4


# d = (1,)
# n = 50
# ux = torch.ones(d).double() * 7
# uy = torch.ones(d).double() * 7
# sx = torch.ones(d).double() * 1.5
# sy = torch.ones(d).double() * 0.5
# r = torch.ones(d).double() * 0.5
# z = cross_correlation(ux, uy, sx, sy, r, n).item()
# print('Matlab:', z - 49.375002608786524)

# n = 10000
# u = torch.tensor([ux, uy]).double()
# s = torch.tensor([[sx**2, r * sx * sy], [r * sx * sy, sy**2]]).double()
# dist = torch.distributions.MultivariateNormal(u, s)
# z_mc = dist.sample([n]).clamp_min_(0).prod(-1).mean().item() * (n / (n - 1))
# print('MC:', z - z_mc)


def main():
    n = 10
    dim = 4
    mu = torch.randn(dim).double()
    cov = rand_cov(dim).double()
    sig = cov.diagonal().sqrt()
    cor = cov / (sig.unsqueeze(-1) @ sig.unsqueeze(0))

    i, j = torch.triu_indices(dim, dim, 1)
    cc = cross_correlation(mu[i], mu[j], sig[i], sig[j], cor[i, j], n)
    print(cc)


if __name__ == "__main__":
    main()
