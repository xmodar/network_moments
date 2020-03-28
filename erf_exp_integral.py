from math import gamma, log, pi, sqrt

import numpy as np
import scipy.integrate
import scipy.special
import torch
from torch.autograd import Function

__all__ = ['erf_exp_integral']


def non_differentiable(function):
    """Decorate a function as non differentiable."""
    name = function.__qualname__

    @staticmethod
    def forward(ctx, *args, **kwargs):  # pylint: disable=unused-argument
        with torch.no_grad():
            return function(*args, **kwargs)

    return type(name, (Function,), {'forward': forward}).apply


@np.vectorize
def numpy_erf_exp_integral(a, b, c):
    """Integrate `erf(a*x+b) * exp(-x**2)` from `c` to infinity in NumPy."""
    f = lambda x: scipy.special.erf(a * x + b) * np.exp(-x**2)
    return scipy.integrate.quad(f, c, np.inf)[0]


@non_differentiable
def torch_erf_exp_integral(a, b, c):
    """Integrate `erf(a*x+b) * exp(-x**2)` from `c` to infinity."""
    cpu = lambda tensor: tensor.detach().cpu().numpy()
    return a.new(numpy_erf_exp_integral(cpu(a), cpu(b), cpu(c)))


class GammaInc(Function):
    """The normalized lower incomplete gamma function.

    Because torch.distributions.Gamma().cdf() is not implemented yet.
    """

    @staticmethod
    def forward(ctx, a, x):  # pylint: disable=arguments-differ
        """Perform the forward pass."""
        ctx.a = a
        ctx.save_for_backward(x)
        return x.new(scipy.special.gammainc(a, x.data.cpu().numpy()))

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        """Perform the backward pass."""
        x, = ctx.saved_tensors
        grad_x = (-x - log(gamma(ctx.a))).exp() * x.pow(ctx.a - 1)
        return None, grad_x * grad_output


gammainc = GammaInc.apply


def polynomial(coefficients, x):
    """Compute a polynomial at `x` given order-decreasing `coefficients`."""
    out = 0
    coefficients = iter(coefficients)
    out += next(coefficients, 0)
    for c in coefficients:
        out *= x
        out += c
    return out


def hermite(n, x):
    """Evaluate the `n`th-degree Hermite polynomial."""
    return polynomial(scipy.special.hermite(n).coeffs.tolist(), x)


def series(a, b, x, n):
    """Sum the first terms in the infinite series from 0 to `n`."""
    a, x, s = a / 2, x * x, -x.sign()

    def term(nu):
        gammas = gammainc(nu, x) / gamma(nu + 0.5)
        return a**(2 * (nu - 0.5)) * hermite(2 * (nu - 1), b) * gammas

    out = 0
    for i in range(n + 1):
        out += term(i + 1) + s * term(i + 1.5)
    return out


def definite_differece(a, b, c, n):
    """Evaluate the erf_exp_integral from `c` to infinity."""
    inf = c.new([float('inf')]).expand(c.size())
    s = series(a, b, inf, n) - series(a, b, c, n)
    return sqrt(pi / 4) * b.erf() * (1 - c.erf()) + (-b * b).exp() * s


def erf_exp_integral(a, b, c, n=-1):
    """Integrate `erf(a*x+b) * exp(-x**2)` from `c` to infinity."""
    if n < 0:
        return torch_erf_exp_integral(a, b, c)
    y = definite_differece(a, b, c, n)
    # handle inaccurate cases
    idx = a.abs() >= 1
    idx, a, b, c = torch.broadcast_tensors(idx, a, b, c)
    a, b, c = a[idx], b[idx], c[idx]
    acb, a_sign = a * c + b, a.sign()
    v = definite_differece(1 / a.abs(), -b / a, acb * a_sign, n)
    v = a_sign * (sqrt(pi / 4) - v) - sqrt(pi / 4) * acb.erf() * c.erf()
    return y.masked_scatter_(idx, v)
