function B = ridge_pen(Y,X)

% Ridge penalization: closed form solution

% inputs:  - Y: d x T vector of responses
%          - X: d x T lagged data matrix       

% output: - B: estimated matrix parameter

lambda = 0.001:0.01:100;
T = size(Y,2); len_in = round(0.75*T); 
X_in = X(:,1:len_in); X_out = X(:,len_in+1:end);
Y_in = Y(:,1:len_in); Y_out = Y(:,len_in+1:end);
theta_fold = zeros(size(Y,1),size(X,1),length(lambda));

parfor jj = 1:length(lambda)
    theta_fold(:,:,jj) = Y_in*X_in'*inv(X_in*X_in'+lambda(jj)*eye(size(X_in,1)));
end
clear jj
count = zeros(length(lambda),1);
for ii = 1:length(lambda)
    count(ii) = sum(sum((Y_out-theta_fold(:,:,ii)*X_out).^2))/(2*length(Y_out));
end
clear ii
[~,ind] = min(count); lambda_opt = lambda(ind);
B = Y*X'*inv(X*X'+lambda_opt*eye(size(X,1)));