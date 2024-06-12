function [lambda_sol,psi_sol] = EM_algorithm(X,m,gamma,mu,Lambda_init,Psi_init)

t = 0.1;
Sy=cov(X);
Lambda=Lambda_init;
Su=Psi_init;
Sigmaold=Su;
Lambda0=Lambda;
A=inv(Lambda0*Lambda0'+Sigmaold);
C=Sy*A*Lambda0;
Eff=eye(m)-Lambda0'*A*Lambda0+Lambda0'*A*C;
Lambda=C*inv(Eff);
Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda';
KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
P=Pmatrix(Su,mu);
B=gamma*t*P;
Sigma1=soft(KML,B);
kk=1;
while  likelihoodTrue(Sy,P,Sigmaold,Lambda0,gamma)- likelihoodTrue(Sy,P,Sigma1,Lambda,gamma)>10^(-7)&kk<5000
    Sigmaold=Sigma1;
    Lambda0=Lambda;  
    A=inv(Lambda0*Lambda0'+Sigma1);
    C=Sy*A*Lambda0;
    Eff=eye(m)-Lambda0'*A*Lambda0+Lambda0'*A*C;
    Lambda=C*inv(Eff);
    Su=Sy-C*Lambda'-Lambda*C'+Lambda*Eff*Lambda';
    KML=Sigmaold-t*(inv(Sigmaold)-inv(Sigmaold)*Su*inv(Sigmaold));
    P=Pmatrix(Su,mu);
    B=gamma*t*P;
    Sigma1=soft(KML,B);
    kk=kk+1;
end
lambda_sol=vec(Lambda0);
psi_sol=vech(Sigmaold);