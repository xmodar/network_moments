function I = e(a, b, c, n)
    x = 1/(2*abs(a));
    y = -b/a;
    z = a*c+b;
    w = sign(sign(a)*z);
    syms u
    I0 = pi/2*w*(2*normcdf(sqrt(2)*abs(z))-1)*(1/2-normcdf(sqrt(2)*b/a));
    s1 = symsum(x^(2*u+1)/gamma(u+3/2)*hermiteH(2*u,y) * (g(u+1,z^2)-1), u, 0, n);
    s2 = symsum(x^(2*u+2)/gamma(u+2)*hermiteH(2*u+1,y) * g(u+3/2,z^2)*w, u, 0, n);
    I2_1 = I0 + sqrt(pi)/2*exp(-y^2) * (s1-s2);
    I2_2 = pi/4 * erf((y)/sqrt(1+(1/a)^2));
    I = sign(a) *(pi/4 + I2_1 - I2_2) - pi/4 *erf(c)*erf(z);
    % I = I * 2 / sqrt(pi);
    double(s2-s1)
    double(I) * 2 / sqrt(pi)
end

