function Hdcc = dcc_for(returns_in,returns_out)

% Inputs:
% - returns_in: T_in x N matrix of observations, with T_in the in-sample
% length and N the number of variables
% - returns_out: T_out x N matrix of observations, with T_out the
% out-of-sample length and N the number of variables

% Output:
% - Hdcc: N x N x T_out variance-covariance process (out-of-sample)

% Note that all the MGARCH models are estimated/generated from the functions
% that where downloaded from the MFE toolbox: please see
% https://www.kevinsheppard.com/code/matlab/mfe-toolbox/
[~,N]=size(returns_in);


% scalar DCC
if (N<101)
    [parameters_dcc,~,H_in]=dcc_mvgarch(returns_in,'full');
else
    [parameters_dcc,~,H_in]=dcc_mvgarch(returns_in,'CL');
end

%%%%%%% Generation of the out-of-sample forecasts for the scalar DCC %%%%%%

h_oos=zeros(size(returns_out,1),size(returns_out,2)); index = 1;
for jj=1:size(returns_out,2)
    univariateparameters=parameters_dcc(index:index+1+1);
    [~, h_oos(:,jj)] = dcc_univariate_simulate(univariateparameters,1,1,returns_out(:,jj));
    index=index+1+1+1;
end
clear jj
h_oos = sqrt(h_oos);
% Rt_dcc_oos: out-of-sample dynamic correlation matrix
[~,Rt_dcc_oos,~,~]=dcc_mvgarch_generate_oos(parameters_dcc,returns_out,returns_in,H_in);

% Hdcc: out-of-sample dynamic variance-covariance matrix, built from
% Rt_dcc_oos and the univariate out-of-sample GARCH(1,1) in h_oos
Hdcc = zeros(N,N,size(returns_out,1)); T = size(returns_out,1);
for t = 1:T
    Hdcc(:,:,t) = diag(h_oos(t,:))*Rt_dcc_oos(:,:,t)*diag(h_oos(t,:));
end

end
