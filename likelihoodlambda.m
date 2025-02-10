function L=likelihoodlambda(S,Lambda,Psi)

Sigma=Lambda*Lambda'+Psi;
p = size(S,2);

L = (log(abs(det(Sigma)))+trace(S/Sigma))/p;

