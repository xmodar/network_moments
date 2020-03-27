function [y] = erf_exp_integral(a, b, c, n)
%ERF_EXP_INTEGRAL Integrate erf(a.*x+b)/exp(x.^2) over x from c to Inf
%   If n >= 0, use the n terms from the infinite sum of the integral.
%   Otherwise, use the best numerical integration method (preferred).

if nargin < 4 || n < 0
    f = @(a, b, c) integral(@(x) erf(a .* x + b) ./ exp(x.^2), c, Inf);
    y = arrayfun(f, a, b, c);
else  % this is implemented for educational purposes only
    % if abs(a) < 1
    y = I(a, b, Inf, n) - I(a, b, c, n);

    % else
    idx = abs(a) >= 1;
    a = a(idx); b = b(idx); c = c(idx);
    upper = I(1./abs(a), -b./a, Inf, n);
    lower = I(1./abs(a), -b./a, (a.*c+b).*sign(a), n);
    y(idx) = (-pi/4).*erf(a.*c+b).*erf(c) + sign(a).*((pi/4)+lower-upper);

    y = y ./ sqrt(pi/4);
end
end

function [y] = I(a, b, x, n)
term1 = (pi / 4) .* erf(x) .* erf(b);
term2 = sqrt(pi / 4) .* exp(-b.^2) .* series(a, b, x, n);
y = term1 + term2;
end

function [y] = series(a, b, x, n)
% syms u
% y = double(symsum(gammaH(u+1,x,a,b)-sign(x).*gammaH(u+1.5,x,a,b),u,0,n));
y = 0;
for u=0:n
    y = y + gammaH(u + 1, x, a, b) - sign(x).*gammaH(u + 1.5, x, a, b);
end
end

function [y] = gammaH(nu, z, a, b)
y = hermiteH(2 .* (nu - 1), b) .* gammainc(nu, z.^2) ./ gamma(nu + 0.5);
y = (a ./ 2) .^ (2 .* (nu - 0.5)) .* y;
end

function [y] = gammainc(nu, z)
y = 1 - igamma(nu, z) ./ gamma(nu);
end
