function [out_covariance] = relu_zero_mean_covariance(in_covariance)
%RELU_ZERO_MEAN_COVARIANCE Output covariance of ReLU for zero-mean Gaussian input.
% f(x) = max(x, 0).
% Args:
%     in_covariance: Input covariance matrix (Size, Size).
% Returns:
%     Output covariance of ReLU for zero-mean Gaussian input (Size, Size).
    v = diag(in_covariance);
    mu = relu_mean(0, v);
    corr = relu_zero_mean_correlation(in_covariance, sqrt(v));
    out_covariance = corr - mu * mu';
end