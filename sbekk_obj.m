function obj = sbekk_obj(theta,mY,mS,method)

% Inputs:
% - theta: 2 x 1 scalar BEKK parameters
% - mY: T x N vector of observations
% - mS: N x N sample variance-covariance matrix
% - method: 'full' for full likelihood; 'CL' for composite likelihood

% Output:
% - obj: loss function

switch method
    case 'full'
        [ll,~] = sbekk_likelihood(theta,mY,mS);
    case 'CL'
        [ll,~] = sbekk_Clikelihood(theta,mY,mS);   
end
if isnan(ll) || isinf(ll) || ~isreal(ll)
    ll = 1e7;
end
obj = ll;
end


