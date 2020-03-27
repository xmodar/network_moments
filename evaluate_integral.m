function I = evaluate_integral(a,b,c,n)
% evaluate integral of ( f = sqrt(pi)/2 * exp(-x.^2) .* erf(a*x+b) )
% should add a 

syms u
if abs(a)<=1
    %a
    %print = '|a| less than a'
    % gammainc  doesn't accept symbolic input
    I0 = pi/2 *sign(c)*(2*normcdf(sqrt(2)*abs(c))-1)*(1/2-normcdf(-sqrt(2)*b));
    s1 = symsum( ((a/2)^(2*u+1))/(gamma(u+3/2)) * (1-igamma(u+1,c^2)/gamma(u+1)) *hermiteH(2*u,b) , u, 0, n);
    s2 = symsum( sign(c)*((a/2)^(2*u+2))/gamma(u+2)* (1-igamma(u+3/2,c^2)/gamma(u+3/2)) *hermiteH(2*u+1,b) , u, 0, n);
    I2_1 = I0 + sqrt(pi)/2 * exp(-b^2) * (s1-s2);
    s3 = symsum(((a/2)^(2*u+1))/(gamma(u+3/2)) * hermiteH(2*u,b), u, 0, n);
    I2_2 = pi/4 *erf(b/sqrt(1+a^2)) + sqrt(pi)/2 *exp(-b^2) * s3;
    I = - I2_1 + I2_2;
end

if abs(a)>1
    I0 = pi/2 *sign( sign(a)*(a*c+b) )*(2*normcdf(sqrt(2)*abs(a*c+b))-1)*(1/2-normcdf(sqrt(2)*b/a));
    s1 = symsum( ((1/(2*abs(a)))^(2*u+1))/(gamma(u+3/2)) * (1-igamma(u+1,(a*c+b)^2)/gamma(u+1)) *hermiteH(2*u,-b/a) , u, 0, n);
    s2 = symsum( sign(sign(a)*(a*c+b))*((1/(2*abs(a)))^(2*u+2))/gamma(u+2)* (1-igamma(u+3/2,(a*c+b)^2)/gamma(u+3/2)) *hermiteH(2*u+1,-b/a) , u, 0, n);
    I2_1 = sign(a)*(I0 + sqrt(pi)/2 * exp(-(b/a)^2) * (s1-s2));
    s3 = symsum(((1/(2*abs(a)))^(2*u+1))/(gamma(u+3/2)) * hermiteH(2*u,-b/a), u, 0, n);
    I2_2 = sign(a)*( pi/4 *erf((-b/a)/sqrt(1+(1/a)^2)) + sqrt(pi)/2 *exp(-(b/a)^2) * s3 );
    I2_x = - I2_1 + I2_2;
    I = pi/4 *sign(a) - pi/4 *erf(c)*erf(a*c+b) - I2_x;
end

I = double(I);

end
