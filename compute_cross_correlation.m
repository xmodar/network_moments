% # this function computes the value of E[max(x,0)*max(y,0)] where x and y are
% # jointly gaussian with the following parameters:
% # means: ux, uy.  standard_deviations: sx, sy.  correlation_coefficient: r
% # one additional parameter "n" is used for the number of terms used for the infinite series
function E = compute_cross_correlation(ux, uy, sx, sy, r, n)

c_expr= (uy^2*sx^2+sy^2*ux^2-2*ux*uy*r*sx*sy)/(2*sx^2*sy^2*(1-r^2));
    a_1 = r/sqrt(1-r^2);
    b_1 = ux/(sqrt(2*(1-r^2))*sx);
    c_1 = -uy/(sqrt(2)*sy);
    I_1 = 2/sqrt(pi) * evaluate_integral(a_1,b_1,c_1,n);
    E =  sx*sy*sqrt(1-r^2)/(2*pi) *exp(-c_expr) + (ux*uy+r*sx*sy)/4 ...
                                + (uy*sx)/(2*sqrt(2*pi)) * exp(-ux^2/(2*sx^2))*(1+erf((uy*sx-r*ux*sy)/(sqrt(2)*sx*sy*sqrt(1-r^2)))) ...
                                + (ux*sy)/(2*sqrt(2*pi)) * exp(-uy^2/(2*sy^2))*(1+erf((ux*sy-r*uy*sx)/(sqrt(2)*sx*sy*sqrt(1-r^2)))) ...
                                + (ux*uy+r*sx*sy)/4 *erf(uy/(sqrt(2)*sy)) + (ux*uy+r*sx*sy)/(2*sqrt(pi)) * I_1 ;
end