function [o_mean, o_covariance] = gaussian_relu_moments(mean, covariance)
%GAUSSIAN_RELU_MOMENTS Get mean and covariance for ReLU(gaussian_vector)
%   We start by Omega from Equation (1) in the Network Moments paper
%   The expression presented in the paper for Omega is for the 2D case
%   (i.e., assuming size(mean) is [2 1] and size(covariance) is [2 2]).
%   Here we will need to generalize this for any number of dimensions n.
%   One approach is to define it for 2D and call it for every possible
%   pair of dimensions, then combine the results in a single matrix.
%   Instead, use [n n] matrices or [n 1] vectors as intermediate resutls.
%   Every element in these matrices naturally corresponds to a pair.
% 	We will be very careful with transposing and broadcasting operations.

n = numel(mean);
assert(all(size(mean) == [n 1]), 'mean must be a column vector');
assert(all(size(covariance) == [n n]), 'covariance must match mean');

%% computing Omega

% det(covariance) could be deceiving since it was written for 2D
% here, we need it for all pairs (symmetric matrix with zero diagonals)
% we can compute sigma_det as std_outer.^2 .* (1 - rho.^2) 
% where rho=covariance./std_outer; correlation coefficients (-1<=rho<=1) or
std = realsqrt(diag(covariance));
std_outer = std' .* std;
sigma_det = max(std_outer - covariance, 0) .* (std_outer + covariance);

% similarly, the term mean' * inv(covariance) * mean is true only in 2D
% here, we can compute this as follows (from the inverse of 2D matrices)
% (std_x_mean'.^2 + std_x_mean.^2 - 2*mean_outer.*covariance) ./ sigma_det
% knowing that, mean_outer .* covariance = std_x_mean .* std_x_mean' .* rho
% note here that the diagonals are undefined for this term since rho = 1
% also, the numerator and denomenator are zero-diagonal symmetric matrices
mean_outer = mean' .* mean;
std_x_mean = mean' .* std;
q = @(x, y) x' + x - 2 .* y;
quadratic = q(std_x_mean.^2, mean_outer .* covariance) ./ sigma_det;

% the first term is symmetric (i.e., term1(x1, x2) == term1(x2, x1))
sqrt_sigma_det = realsqrt(sigma_det);
term1 = (1 / (2 * pi)) .* sqrt_sigma_det .* exp(-0.5 .* quadratic);

% sigma_tilde_mean = sigma_tilde * mean / sqrt(2) (in 2D, sigma_tilde is):
% diag(sqrt(det(cov) ./ [cov(2, 2); cov(1, 1)])) * inv(cov)
% but here, the result is a non-symmetric matrix with undefined diagonals
mean_std = sqrt(1 / 2) .* mean ./ std;
sigma_tilde_mean = sqrt(1 / 2) .* std_x_mean' - mean_std' .* covariance;
sigma_tilde_mean = sigma_tilde_mean ./ sqrt_sigma_det;

% the second and third terms is an undefined-diagonal non-symmetric matrix
% because of this asymmetry, it is a bit tricky (i.e., pairs are ordered)
pdf = (1 / sqrt(2 * pi)) .* exp(-mean_std.^2);
term2 = (1 / 2) .* std_x_mean' .* pdf' .* (1 + erf(sigma_tilde_mean));

% the forth term is also non-symmetric because of cdf'
cdf = (1 / 2) .* (1 + erf(mean_std));
mean_sigma = mean_outer + covariance;
term4 = (1 / 2) .* mean_sigma .* cdf';

% now to the final result Omega
omega = term1 + term2 + term2' + term4;

%% integrating erf(a*x+b)/exp(x^2) from c to Inf

% a = rho .* irho; b = mean_std .* irho; c = -mean_std';
irho = std_outer ./ sqrt_sigma_det;  % equivalent to 1 ./ sqrt(1 - rho.^2)
a = irho .* covariance ./ std_outer;
b = irho .* mean_std;
c = repmat(-mean_std', n, 1);
b(irho == Inf) = 0;  % avoid Inf-Inf in a .* x + b (when irho is Inf)
f = @(a, b, c) integral(@(x) erf(a .* x + b) ./ exp(x.^2), c, Inf);
integration = arrayfun(f, a, b, c);  % erf_exp_integral(a, b, c);

%% correlation matrix E[ReLU(x)ReLU(x)^T]

correlation = omega + (1 / sqrt(4 * pi)) .* mean_sigma .* integration;
second_moment = diag(mean_sigma) .* cdf + diag(std_x_mean) .* pdf;
correlation(1:n + 1:end) = second_moment;  % replace the diagonal

%% mean and covariance matrix E[ReLU(x)ReLU(x)^T] - E[ReLU(x)]E[ReLU(x)]^T

o_mean = mean .* cdf + std .* pdf;
o_covariance = correlation - o_mean' .* o_mean;

end