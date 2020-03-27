close all; clear; clc;

% rng(42);

n = 20;
mu = randn(n, 1);
q = qr(randn(n, n));
sigma = q * q';
clear q;

[o_mu, o_sigma] = gaussian_relu_moments(mu, sigma);
[mc_mu, mc_sigma] = moments(max(mvnrnd(mu, sigma, 10000000), 0));

disp('mu');    disp(similar(mc_mu,    o_mu,    1e-8, 1e-2));
disp('sigma'); disp(similar(mc_sigma, o_sigma, 1e-8, 1e-2));


function [mu, sigma] = moments(samples)
mu = mean(samples, 1)';
sigma = cov(samples);
end

function [out] = similar(a, b, rtol, atol)
% disp(a)
% disp(b)
% disp(norm(a(:) - b(:)))
% disp(mean(abs(a(:) - b(:))))
if nargin < 3
    rtol = 1e-5;
end
if nargin < 4
    atol = 1e-8;
end
out = abs(a - b) <= atol + rtol .* abs(b);
if all(out(:))
    out = true;
end
end

