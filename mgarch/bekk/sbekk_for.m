function Hsbekk = sbekk_for(returns_in,returns_out)

% Inputs:
% - returns_in: T_in x N matrix of observations, with T_in the in-sample
% length and N the number of variables
% - returns_out: T_out x N matrix of observations, with T_out the
% out-of-sample length and N the number of variables

% Output:
% - Hsbekk: N x N x T_out variance-covariance process (out-of-sample)

%%%%%%%%%%%%%%%% Estimation of the scalar BEKK model %%%%%%%%%%%%%%%%%
% VT-estimation
[parameters_sbekk,~] = sbekk(returns_in);
mL = unvech(parameters_sbekk(1:end-2));
mS = mL*mL';
theta = parameters_sbekk(end-1:end);
[~,Hsbekk] = sbekk_likelihood(theta,returns_out,mS);
end
