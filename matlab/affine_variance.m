function [out_variance] = affine_variance(in_covariance, A)
%AFFINE_VARIANCE Output covariance matrix of Affine for general input.
% f(x) = A * x + b.
% Args:
%     in_covariance: Input covariance matrix of size (Size, Size)
%         or variance of size (Size).
%     A: Matrix (M, Size).
% Returns:
%     Output variance of Affine for general input (M, M).
    out_variance = sum((A * in_covariance) .* A, 2);
end

