function [parameters_sbekk,mH] = sbekk(mY)

% Inputs:
% - mY: T x N vector of observations

% Outputs:
% - parameters_sbekk: estimated scalar BEKK parameters
% - mG: N x N x T variance-covariance process based on the estimated scalar
% BEKK model

optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 300000;
optimoptions.MaxFunEvals = 300000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 300000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

mS = cov(mY);

vTheta0 = [0.1;0.2];

if (size(mY,2)<101)
    method = 'full';
else
    method = 'CL';
end
[param,~] = fmincon(@(x)sbekk_obj(x,mY,mS,method),vTheta0,[],[],[],[],[],[],@(x)sbekk_nlcon(x),optimoptions);
[~,mH] = sbekk_likelihood(param,mY,mS);
parameters_sbekk = [vech(chol(mS,"lower"));param];
end