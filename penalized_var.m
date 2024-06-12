function [b_est,lambda_opt] = penalized_var(Y,X,lambda)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - lambda: vector of candidates for the tuning parameter (user
%           specified)
%         - method: 'lasso', 'alasso', 'scad' or 'mcp'
%           'alasso' stands for adaptive LASSO
%           'lasso' and 'alasso' are solved by the Shooting algorithm
%           'scad' and 'mcp' are numerically solved by fmincon

% Outputs:
%         - b_est: vector of estimated parameters
%         - lambda_opt: optimal tuning parameter selected by
%           cross-validation for the corresponding penalty function

[T,p] = size(X); len_in = round(0.75*T);
X_in = X(1:len_in,:); X_out = X(len_in+1:end,:);
Y_in = Y(1:len_in,:); Y_out = Y(len_in+1:end,:);
theta_fold = zeros(p,length(lambda));
parfor jj = 1:length(lambda)
    theta_fold(:,jj) = lassoShooting(Y_in,X_in,lambda(jj),1); 
end
clear jj

count = zeros(length(lambda),1);
for ii = 1:length(lambda)
    count(ii) = sum((Y_out-X_out*theta_fold(:,ii)).^2)/(2*length(Y_out));
end
clear ii
[~,ind] = min(count); lambda_opt = lambda(ind);
b_est = lassoShooting(Y,X,lambda_opt,1);