function z = f(x, y)
    I = @(a, b, c) integral(@(x) erf(a.*x+b)./exp(x.^2), c, Inf);
    ix = sqrt(1-x.^2);
    iy = sqrt(0.5)*y;
    z = arrayfun(I, x./ix, iy./ix, -iy);
end