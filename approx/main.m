close all; clear; clc;

terms = 10;
lin = linspace(-0.3, 0.3, 50);
[m_s, rho] = meshgrid(10*lin, lin);
out = series(m_s, rho, terms);

%%
surf(m_s, rho, out);

% load_file = true;
% if load_file && isfile('main.mat')
%     clear load_file; load('main.mat');
% else
%     lin = linspace(-1+1e-5, 1-1e-5, 101);
%     [rho, m_s] = meshgrid(lin, lin.*10);
%     out = fun(rho, m_s);
%     clear lin load_file; save('main.mat');
% end
% com = approximation(rho, m_s);

% figure; surf(rho, m_s, out);
% figure; surf(rho, m_s, com);
% figure; surf(rho, m_s, out-com);
% 
% function z = fun(x, y)
%     I = @(a, b, c) integral(@(x) erf(a.*x+b)./exp(x.^2), c, Inf);
%     ix = sqrt(1-x.^2);
%     iy = sqrt(0.5)*y;
%     z = arrayfun(I, x./ix, iy./ix, -iy);
% end
% 
% function z = approximation(x, y)
%     s = (x + 1)./2;
%     b = erf(2*s.^sqrt(1/pi)).*sqrt(1/2)+(2-sqrt(2)/2);
%     d = s.^(1/(2/sqrt(pi)+1)).*(sqrt(2)-sqrt(pi)/2)+sqrt(pi)/2;
%     z = sqrt(pi)*normcdf(y)-acos(x).*exp(-abs(y./d).^b)/sqrt(pi);
% end