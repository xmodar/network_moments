function [out_correlation] = relu_zero_mean_correlation(in_covariance, s)
%RELU_ZERO_MEAN_CORRELATION Output correlation of ReLU for zero-mean Gaussian input.
% f(x) = max(x, 0).
% Args:
%     in_covariance: Input covariance matrix (Size, Size).
% Returns:
%     Output correlation of ReLU for zero-mean Gaussian input (Size, Size).
    if ~exist('s','var')
        s = sqrt(diag(in_covariance));
    end
    S = s * s';
    V = min(max(in_covariance ./ S, -1), 1);
    temp1 = in_covariance .* asin(V);
    temp2 = S .* sqrt(1 - (V .^ 2));
    out_correlation = (temp1 + temp2) / (2 * pi) + in_covariance / 4;
end