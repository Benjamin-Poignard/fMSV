function [Psi,param_psi] = psi_estimation(X,m,Lambda,Psi_init)

% Inputs:
%          - X: n x p matrix of observations
%          - m: number of factors (a priori set by the user)
%          - Lambda: p x m sparse factor loading matrix
%          - Psi_init: initial parameter of the diagonal
%          variance-covariance matrix  of the idiosyncratic errors
% Output:
%          - Psi: diagonal variance-covariance matrix  of the idiosyncratic
%          errors obtained by the EM algorithm

[T,N]=size(X);
Sy=X'*X/T;
kk=1;
Psi=Psi_init;
Psi_old=eye(N)*100;

while likelihoodlambda(Sy,Lambda,Psi_old)-likelihoodlambda(Sy,Lambda,Psi)>10^(-7)&&kk<4000
    Psi_old=Psi;
    A=inv(Lambda*Lambda'+Psi);
    C=Sy*A*Lambda;
    Eff=eye(m)-Lambda'*A*Lambda+Lambda'*A*C;
    M=Sy-Lambda*Lambda'*A*Sy;
    Psi=diag(diag(M));
    kk=kk+1 ;
end
param_psi = diag(Psi);