close all; clear; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing the mean of the following function:
% B * max(A*x+c1,0) + c2 , 0) where A, B  [could be of any size] 
% NN: Affine >> Relu >> Affine
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = 25; %number of variables
m = 10000; %number of montecarlo runs
mu = randn(n,1); %means of input joint gaussin
dummy = rand(n,n);
[U, ~] = eig((dummy'+dummy)/2); %summetric matrix
in_covariance = U * diag(abs(rand(n,1)) + 5*abs(rand(n,1)))* U'; %PSD matrix
samples = mvnrnd(mu,in_covariance,m')'; %sample data

%Affine functions
k = 2;
p = 2;
A = 10*randn(k,n); %linear factors co-offecients (Lienar transform)
B = randn(p,k); %linear factors co-offecients (Lienar transform)
c1 = repmat(100*randn(k,1),[1,m]);
c2 = repmat(100*randn(p,1),[1,m]);

%% MonteCarlo/Analytic Mean
simulation_mean = mean(B * max(A * samples + c1, 0) + c2, 2);
analytic_mean = affine_relu_affine_mean(mu, in_covariance, A, c1(:,1), B, c2(:,1));
fprintf('MonteCarlo Mean --- Analytic Mean: ')
simulation_mean ./ analytic_mean

%% MonteCarlo/Analytic Variance
c1 = repmat(-A*mu,[1,m]);
simulation_variance = var(B * max(A * samples + c1, 0) + c2,[],2);
analytic_variance = affine_relu_affine_variance(in_covariance, A, B);
fprintf('MonteCarlo Variance --- Analytic Variance: ')
simulation_variance ./ analytic_variance