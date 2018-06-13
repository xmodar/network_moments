function [out_mean] = relu_mean(in_mean, in_variance)
%RELU_MEAN Output mean of ReLU for general Gaussian input.
% f(x) = max(x, 0).
% Args:
%     in_mean: Input mean of size (Batch, Size).
%     in_variance: Input variance vector (Batch, Size)
%         or scalar v such that variance = v * ones(Size).
% Returns:
%     Output mean of ReLU for general Gaussian input (Batch, Size).
    s = sqrt(in_variance);
    temp1 = s / sqrt(2 * pi);
    if sum(abs(in_mean(:))) == 0
        out_mean = temp1;
    else
        u = in_mean ./ (sqrt(2) * s);
        temp2 = 0.5 * in_mean .* (1 + erf(u));
        out_mean = temp1 .* exp(-(u .^ 2)) + temp2;
    end
end

