function [Lambda,gamma_opt,Psi] = sparse_factor_TS(X,m,loss,gamma,method,Lambda_first,Psi_first)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - loss: 'Gaussian' or 'LS' ('LS' stands for least squares)
%          - gamma: tuning parameter (grid of candidates set by the user)
%          - method: SCAD or MCP penalization (a_scad = 3.7, b_mcp = 3.5):
%          see lambda_penalized.m to modify a_scad and b_mcp
%          - Lambda_first (optional input): inital parameter value for the
%          factor loading matrix
%          - Psi_first (optional input): inital parameter value for the
%          variance-covariance matrix (diagonal) of the idiosyncratic
%          errors, jointly obtained with Lambda_first
% Outputs:
%          - Lambda: sparse factor loading matrix
%          - gamma_opt: optimal tuning parameter selected by the
%          out-of-sample cross-validation procedure
%          - Psi: variance-covariance matrix (diagonal) of the
%          idiosyncratic errors

% if no first step estimator for Lambda and Psi are provided, then the
% initial values run the following function to get an initial point
if nargin < 6
    [Lambda_first,Psi_first] = non_penalized_factor(cov(X),m,loss);
end

p = size(X,2);

% Lambda-step estimation
[Lambda_rotated,~] = penalized_factor_Qstep(Lambda_first,m,gamma,method);
[Lambda_step,gamma_opt] = lambda_penalized_TS(X,m,Lambda_rotated,Psi_first,loss,gamma,method);

% Psi-step estimation
Psi_step = psi_estimation(X,m,Lambda_step,Psi_first,loss);

param_psi_update = diag(Psi_step); param_lambda_update = vec(Lambda_step);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Iterate until convergence %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Tol = 10^(-5); count = 0; max_iter = 500;
while count < max_iter
    
    count = count+1;
    param_psi = param_psi_update; param_lambda = param_lambda_update;
    
    % Lambda-step
    if length(gamma)>1
        Lambda = zeros(p,m,length(gamma));
        for i=1:length(gamma)
            Lambda(:,:,i) = reshape(param_lambda,p,m);
        end
    else
        Lambda = reshape(param_lambda,p,m);
    end
    [Lambda_step,gamma_opt] = lambda_penalized_TS(X,m,Lambda,diag(param_psi),loss,gamma,method);
    lambda_step = vec(Lambda_step);
    
    % Psi-step
    Psi_step = psi_estimation(X,m,Lambda_step,diag(param_psi),loss);
    
    param_psi_update = diag(Psi_step); param_lambda_update = lambda_step;
    
    if (norm([param_psi_update;param_lambda_update] - [param_psi;param_lambda])/max([1,norm([param_psi_update;param_lambda_update]),norm([param_psi;param_lambda])]) <= Tol)
        break
    end
    
end
Psi = diag(param_psi_update); Lambda = reshape(param_lambda_update,p,m);
switch loss
    case 'Gaussian'
        switch method
            case 'scad'
                fprintf(1,'Estimation with scad-penalized Gaussian loss completed \n')
            case 'mcp'
                fprintf(1,'Estimation with mcp-penalized Gaussian loss completed \n')
        end
    case 'LS'
        switch method
            case 'scad'
                fprintf(1,'Estimation with scad-penalized LS loss completed \n')
            case 'mcp'
                fprintf(1,'Estimation with mcp-penalized LS loss completed \n')
        end
end