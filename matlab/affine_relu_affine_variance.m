function [out_variance] = affine_relu_affine_variance(in_covariance, A, B)
%AFFINE_RELU_AFFINE_VARIANCE Output variance of Affine-ReLU-Affine for special Gaussian input.
% f(x) = B*max(A*x+c1, 0)+c2, where c1 = -A*input_mean.
% For this specific c1, this function doesn't depend on
% neither the input mean nor the biases.
% Args:
%     in_covariance: Input covariance matrix (Size, Size).
%     A: The A matrix (M, Size).
%     B: The B matrix (N, M).
% Returns:
%     Output variance of Affine-ReLU-Affine for Gaussian input
%     with mean = `in_mean` and covariance matrix = `in_covariance`
%     where the bias of the first affine = -A*`in_mean`.
    a_covariance = affine_covariance(in_covariance, A);
    r_covariance = relu_zero_mean_covariance(a_covariance);
    out_variance = affine_variance(r_covariance, B);
end

