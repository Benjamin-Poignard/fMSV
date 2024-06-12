function obj = sbekk_obj(theta,mY,mS,method)
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


