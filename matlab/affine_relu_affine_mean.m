function [out_mean] = affine_relu_affine_mean(in_mean,in_covariance, A, c1, B, c2)
%AFFINE_RELU_AFFINE_MEAN Output mean of Affine-ReLU-Affine for general Gaussian input.
% f(x) = B*max(A*x+c1, 0)+c2.
% Args:
%     in_mean: Input mean of size (Batch, Size).
%     in_covariance: Input covariance matrix (Size, Size).
%     A: The A matrix (M, Size).
%     c1: The c1 vector (M).
%     B: The B matrix (N, M).
%     c2: The c2 vector (N).
% Returns:
%     Output mean of Affine-ReLU-Affine for Gaussian input
%     with mean = `in_mean` and covariance matrix = `in_covariance`.
    a_mean = affine_mean(in_mean, A, c1);
    a_variance = affine_variance(in_covariance, A);
    r_mean = relu_mean(a_mean, a_variance);
    out_mean = affine_mean(r_mean, B, c2);
end

