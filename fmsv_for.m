function [Hf_fmsv,m_opt] = fmsv_for(X_in,X_out,p,m,method,fscore)

% Inputs
% - X_in: T_in x N matrix of observations, with T_in the in-sample length
% and N the number of variables
% - X_out: T_out x N matrix of observations, with T_out the out-of-sample
% length and N the number of variables
% - p: number of lags at at the first step MSV estimation (VAR(p))
% - m: number of factors
% - method: estimation method of the factor model Lambda'*Lambda + Psi
%   - 'scad': sparse Lambda estimation with scad, and Psi diagonal
%   - 'mcp': sparse Lambda estimation with mcp, and Psi diagonal
%   - 'saf': sparse Psi estimation with adaptive Lasso, non-sparse Lambda,
%   under the restriction Lambda'*inv(Psi)*Lambda diagonal
% - fscore: estimator of the underlying factors:
%   - 'WLS': linear-based estimator
%   - 'Gaussian': Gaussian-based estimator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% First step: Factor model estimation %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variance-covariance matrix deduced from the factor model representation
% Sigma = Lambda*Lambda'+ Psi;
[T_in,N] =size(X_in);
% grid: user-specified grid of values for cross-validation
grid = m*[0.01 0.015:0.05:6];


switch method
    
    case 'scad'
        
        gamma = grid*sqrt(log(N*m)/T_in);
        [Lambda_g,Psi_g] = non_penalized_factor(X_in,m,'Gaussian');
        [Lambda,~,Psi] = sparse_factor_TS(X_in,m,'Gaussian',gamma,method,Lambda_g,Psi_g);
        
    case 'mcp'
        
        gamma = grid*sqrt(log(N*m)/T_in);
        [Lambda_g,Psi_g] = non_penalized_factor(X_in,m,'Gaussian');
        [Lambda,~,Psi] = sparse_factor_TS(X_in,m,'Gaussian',gamma,method,Lambda_g,Psi_g);
        
    case 'SAF'
        
        gamma = grid*sqrt(log(N*(N+1)/2)/T_in);
        [Lambda,~,Psi] = approx_factor_TS(X_in,m,gamma);
        
end

%%% Filtering of the factors
iI=[];
for i=1:m
    if (Lambda(:,i)==zeros(N,1))
        iI=[iI i];
    end
end
Lambda(:,iI)=[];
[~,m_opt] = size(Lambda);

switch fscore
    case 'WLS'
        % weghted least squares estimator
        F_in = (X_in*inv(Psi)*Lambda)*inv(Lambda'*inv(Psi)*Lambda);
        F_out = (X_out*inv(Psi)*Lambda)*inv(Lambda'*inv(Psi)*Lambda);
    case 'Gaussian'
        % Gaussian estimator
        F_in = (X_in*inv(Psi)*Lambda)*inv(eye(m_opt)+Lambda'*inv(Psi)*Lambda);
        F_out = (X_out*inv(Psi)*Lambda)*inv(eye(m_opt)+Lambda'*inv(Psi)*Lambda);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% Second step: Estimation of MSV %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ts = size(X_out,1);
Hf_fmsv = zeros(N,N,Ts);

if (m_opt==0)
    for t = 1:Ts
        Hf_fmsv(:,:,t) = Psi;
    end
else
    H_msv_ols_alasso = msv_corrected_for(F_in,F_out,'pen',p);
    for t = 1:Ts
        Hf_fmsv(:,:,t) = Lambda*reshape(H_msv_ols_alasso(:,:,t),[m_opt m_opt])*Lambda' + Psi;
    end
end

end