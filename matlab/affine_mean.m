function [out_mean] = affine_mean(in_mean, A, b)
%AFFINE_MEAN Output mean of Affine for general input.
% f(x) = A * x + b.
% Args:
%     in_mean: Input mean of size (Batch, Size).
%     A: Matrix of size (M, Size).
%     b: bias vector of size (M).
% Returns:
%     Output mean of Affine for general input (Batch, M).
    if isvector(in_mean) && iscolumn(in_mean)
        out_mean =  A * in_mean + b;
    else
        out_mean = in_mean * A' + b;
    end
end

