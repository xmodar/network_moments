'''Network Moments (NMs).

Let x be a random variable with some mean M and covariance S.
x can be multivariate of size (N), so S of size (N, N) and M of size (N).
S = C - outer_product(M, M), where C is the correlation matrix.
M and C are the expectations of x and outer_product(x, x), respectively.
The n-th moment of x is the expectation of x_to_the_power_n.
The diagonal of S and C are the variance and second_moment, respectively.
The second_moment is the expectation of x_squared.
The variance = second_moment - M_squared.

For any function acting on x (e.g., f(x)),
we want to compute its probability density function (i.e., of f(x)).
A simpler task maybe is to find the n-th-moment of the function for all n > 0.

This package is trying to find closed form expressions for the output
probabilistic moments of some functions given some input distributions.
'''
from . import affine
from .. import (utils, net)
