function mC = fNCov(mA)
  % nearest covariance matrix
  [mP,mE] = eig(mA);
  vE = diag(mE);iM=length(vE);
  vEs = vE.*(vE>0) + (1e-4)*ones(iM,1).*(vE<=0);
  mC = mP*diag(vEs)*mP';
  %mC = mP*abs(mE)*mP';
end