function out = series(mean_std, rho, terms)
irho = sqrt(1-rho.^2);
a = rho./irho;
c = mean_std./sqrt(2);
b = c./irho;
[w, x, y, z] = variables(a, b, -c);

% syms u;
% out = symsum(term(u, sign(w), x, y, z.^2), u, 0, terms);
% out = double(out);
out = 0;
w = sign(w);
z = z.^2;
for u=0:terms
    out = out + term(u, w, x, y, z);
end

end

function [w, x, y, z] = variables(a, b, c)
x = a./2;
z = c;
y = b;
w = c;
condition = abs(a)>1;
x_ = 1./(2.*abs(a));
z_ = a.*c+b;
y_ = -b./a;
w_ = a.*z;
x(condition) = x_(condition);
y(condition) = y_(condition);
z(condition) = z_(condition);
w(condition) = w_(condition);
end

function out = term(u, w, x, y, z)
s0 = value(2*u+1, 2*u+2, u+2, x, y);
s1 = value(2*u, 2*u+1, u+3/2, x, y);
f0 = w.*gammainc(u+3/2, z);
f1 = 1-gammainc(u+1, z);
out = f0.*s0+f1.*s1;
end

function out = value(h, e, d, x, y)
out = (x.^e).*hermiteH(h, y)./gamma(d);
end

function out = gammainc(nu, z)
out = 1-igamma(nu, z)./gamma(nu);
end
