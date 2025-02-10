function [Hf_fmsv,m_opt] = fmsv_for(X_in,X_out,p,m,method,fscore)

% Inputs
% - X_in: T_in x N matrix of observations, with T_in the in-sample length
% and N the number of variables
% - X_out: T_out x N matrix of observations, with T_out the out-of-sample
% length and N the number of variables
% - p: number of lags at at the first step MSV estimation (VAR(p))
% - m: number of factors
% - method: estimation method of the factor model Lambda'*Lambda + Psi
%   - 'SFM': sparse Lambda estimation with scad/Psi diagonal Gaussian loss
%   - 'POET': sparse Psi estimation with POET, non-sparse Lambda obtained
%   by PCA
% - fscore: estimator of the underlying factors:
%   - 'WLS': linear-based estimator
%   - 'Gaussian': Gaussian-based estimator

% Outputs:
% - Hf_fmsv: N x N x T factor model-based stochastic variance covariance
% matrix (filtered by MMSLE)
% - m_opt: number of active factors

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% First step: Factor model estimation %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Variance-covariance matrix deduced from the factor model representation
% Sigma = Lambda*Lambda'+ Psi;
[T_in,N] =size(X_in);

switch method
    case 'SFM'
        gamma = m*[0.001:0.001:0.01 0.011:0.1:6]*sqrt(log(N*m)/T_in);
        [Lambda_g,Psi_g] = non_penalized_factor(X_in,m);
        [Lambda,~,Psi] = sparse_factor_TS(X_in,m,gamma,'scad',Lambda_g,Psi_g); % set 'mcp' in lieu of 'scad' for MCP regularization
    case 'POET'
        [~,Lambda,Psi] = poet(X_in',m,0.8,'soft');
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
Ts = size(X_out,1); Hf_fmsv = zeros(N,N,Ts);
if (m_opt==0)
    for t = 1:Ts
        Hf_fmsv(:,:,t) = Psi;
    end
else
    H_msv_ols_alasso = msv_corrected_for(F_in,F_out,p);
    for t = 1:Ts
        Hf_fmsv(:,:,t) = Lambda*reshape(H_msv_ols_alasso(:,:,t),[m_opt m_opt])*Lambda' + Psi;
    end
end
end