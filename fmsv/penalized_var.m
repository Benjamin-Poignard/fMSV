function [b_est,lambda_opt] = penalized_var(Y,X)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables

% Outputs:
%         - b_est: vector of estimated parameters
%         - lambda_opt: optimal tuning parameter selected by
%           cross-validation for the corresponding penalty function

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Step 1: LASSO Estimation %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[T,~] = size(X); len_in = round(0.75*T);
X_in = X(1:len_in,:); X_out = X(len_in+1:end,:);
Y_in = Y(1:len_in,:); Y_out = Y(len_in+1:end,:);
[theta_fold,stats] = lasso(X_in,Y_in);
loss = zeros(size(theta_fold,2),1);
for ii = 1:length(loss)
    loss(ii) = sum((Y_out-X_out*theta_fold(:,ii)).^2)/(2*length(Y_out));
end
clear ii
[~,ind] = min(loss); lambda_cand = stats.Lambda; lambda_opt = lambda_cand(ind);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%% Step 2: Adaptive LASSO Estimation %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

grid = (0.001:0.1:5); lambda = grid*(lambda_opt);
theta_fold = zeros(size(X_in,2),length(lambda));
parfor jj = 1:length(lambda)
    theta_fold(:,jj) = Rlasso_cvx(X_in,Y_in,lambda(jj));
end
clear jj

loss = zeros(length(lambda),1);
for ii = 1:length(lambda)
    loss(ii) = sum((Y_out-X_out*theta_fold(:,ii)).^2)/(2*length(Y_out));
end
clear ii
[~,ind] = min(loss); lambda_opt = lambda(ind);
b_est = Rlasso_cvx(X,Y,lambda_opt);
if sum(b_est)==0
    b_est = Rlasso_cvx(X,Y,lambda(1));
end