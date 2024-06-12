function Hsbekk = sbekk_for(returns_in,returns_out)
%%%%%%%%%%%%%%%% Estimation of the scalar BEKK model %%%%%%%%%%%%%%%%%
% VT-estimation
[parameters_sbekk,~] = sbekk(returns_in);
mL = unvech(parameters_sbekk(1:end-2));
mS = mL*mL';
theta = parameters_sbekk(end-1:end);
[~,Hsbekk] = sbekk_likelihood(theta,returns_out,mS);
end
