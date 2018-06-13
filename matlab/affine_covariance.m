function [out_covariance] = affine_covariance(in_covariance, A)
%AFFINE_COVARIANCE Output covariance matrix of Affine for general input.
% f(x) = A * x + b.
% Args:
%     in_covariance: Input covariance matrix of size (Size, Size)
%         or variance of size (Size).
%     A: Matrix (M, Size).
% Returns:
%     Output covariance of Affine for general input (M, M).
    out_covariance = A * in_covariance * A';
end

