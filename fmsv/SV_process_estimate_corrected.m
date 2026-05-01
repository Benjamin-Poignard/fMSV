function [b,B_hat,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate_corrected(data,p)

% MSV parameters estimation

% Inputs:
%        - data: T x N vector of observations
%        - p: number of lags for the first step
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
% x = log(data.^2)';
iC = 1e-4; v = mean(data.^2);
x = log(data.^2+ iC*v)' - (iC*(ones(T,1)*v)./(data.^2+ iC*v))';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% First step: penalization %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Equation by equation penalisation
%%% Penalisation is performed for the adaptive lasso

% creation of the vector of covariate
X = []; Xnc = []; xx = x-mean(x,2);
for tt = p+1:T
    x_temp_reg = []; x_temp_reg_nc = [];
    for kk = 1:p
        x_temp_reg = [x_temp_reg ; xx(:,tt-kk)];
        x_temp_reg_nc = [x_temp_reg_nc ; x(:,tt-kk)];
    end
    X = [X , x_temp_reg]; Xnc = [Xnc , x_temp_reg_nc];
end
YY = x(:,p+1:end); y = xx'; XX = X';

% equation-by-equation penalized estimation procedure (with targeting)
b = zeros(N,1+p*N); Xbar = mean(x,2); y_obj = y(p+1:end,:);
for ii = 1:N
    [b_alasso,~] = penalized_var(y_obj(:,ii),XX);
    b(ii,:) = [(1-sum(b_alasso))*Xbar(ii),b_alasso'];
end
XX = [ones(T-p,1),Xnc'];
% B corresponds to the estimated sparse Psi matrix in step 1
B = b;

% obtain the residuals and variance covariance
u = YY-B*XX'; Tu = length(u);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Second step: OLS estimation %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    B_hat = ridge_pen(Y_second,XX);
else
    B_hat = Y_second*XX'*inv(XX*XX');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Third step: correlation matrix %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gamma = corr(data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%% Recover the MSV parameters %%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Sig_x = cov(x(:,p+1:end)');
rsb = 0.5*(pi^2)/mean(diag(Sig_x));
rsb = rsb*(rsb<1) + 0.9999*(rsb>1);% adjustment for not exceeding 1
Sig_alpha = (1-rsb)*Sig_x;
Sig_zeta = rsb*Sig_x;
