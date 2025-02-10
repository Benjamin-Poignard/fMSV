function [b,B_hat,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate_corrected(data,p,lambda)

% This code estimates the MSV parameters only
% Use generate_SV_process.m to generate the MSV based variance covariance
% process

% Inputs:
%        - data: T x N vector of observations
%        - p: number of lags for the first step
%        - lambda: penalisation parameter
% Outputs:
%        - b: first step estimator
%        - B_hat: second step estimator
%        - Sig_zeta and Sig_alpha: please refers to the paper for the
%          definitions of these quantities, which correspond to
%          \Sigma_\zeta and \Sigma_\alpha
%        - Gamma: correlation estimator obtained in the third step

% T: number of simulated observations; N: dimension of the vector
[T,N] = size(data);

%%%%%% define the vectors and matrix %%%%%%
% Sigma is the true variance covariance matrix
% x corresponds to the log(data^2)
x = log(data.^2)';

%%%%%%%%%%%%%%%%%%%%%%%%% First step: penalisation %%%%%%%%%%%%%%%%%%%%%%%%

%%% Equation by equation penalisation: this will be useful when considering
%%% large N
%%% Penalisation is performed for the adaptive lasso

% creation of the vector of covariate
X = [];
for tt = p+1:T
    x_temp_reg = [];
    for kk = 1:p
        x_temp_reg = [x_temp_reg ; x(:,tt-kk)];
    end
    X = [X , x_temp_reg];
end
YY = x(:,p+1:end); y = x';

XX = [ones(T-p,1),X']; b = zeros(N,1+p*N);
y = y - ones(length(y),1)*mean(y);

% equation-by-equation penalized estimation procedure
for ii = 1:N
    
    [b_alasso,~] = penalized_var(y(p+1:end,ii),XX,lambda);
    b(ii,:) = b_alasso';
    % non-penalized model: simple OLS
    b_np = ((XX'*XX)\(XX'*y(p+1:end,ii)))';
    b(ii,:) = b_np;
end

% B corresponds to the estimated sparse Psi matrix in step 1
B = b;

% obtain the residuals and variance covariance
u = YY-B*XX'; Tu = length(u);

%%%%%%%%%%%%%%%%%%%%%%% Second step: OLS estimation %%%%%%%%%%%%%%%%%%%%%%%

% creation of the vector of covariates
x_second = x(:,p+1:end);
XX = [];
for tt = 2:Tu
    XX = [XX  [1;x_second(:,tt-1);u(:,tt-1)]];
end
% OLS estimator second step
Y_second = x_second(:,2:end); % dependent variable
if min(eig(XX*XX'))<0.0001
    % Apply Ridge regularization
    B_hat = Y_second*XX'*inv(XX*XX'+0.01*eye(size(XX,1)));
else
    B_hat = Y_second*XX'*inv(XX*XX');
end
%%%%%%%%%%%%%%%%%%%%%% Third step: correlation matrix %%%%%%%%%%%%%%%%%%%%%

Gamma = corr(data);

%%%%%%%%%%%%%%%%%%%%%%%% Recover the MSV parameters %%%%%%%%%%%%%%%%%%%%%%%

Sig_x = cov(x(:,p+1:end)');
rsb = 0.5*(pi^2)/mean(diag(Sig_x));
rsb = rsb*(rsb<1) + 0.9999*(rsb>1);% adjustment for not exceeding 1
Sig_alpha = (1-rsb)*Sig_x;
Sig_zeta = rsb*Sig_x;
