function out = s(a, b, c, d, n)
    out = 0;
    for u = 0:n-1
        f0 = sign(d) * gammainc(c^2, u+3/2)
        f1 = 1 - gammainc(c^2, u+1)
        s0 = (a^(2*u+2))/gamma(u+2)*hermiteH(2*u+1,b)
        s1 = (a^(2*u+1))/gamma(u+3/2)*hermiteH(2*u,b)
        out = out + f0 * s0 + f1 * s1;
    end
end