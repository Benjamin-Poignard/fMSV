function mC = HAC(vE,mX)
[iT,iK]=size(mX);
mV = (vE*ones(1,iK)).*mX;
%iH=floor(iT^(1/5));iH
iH=25;
mC = (1/iT)*(mV'*mV);
for i=1:iH
    mC1 = (1/iT)*mV(1:(end-i+1),:)'*mV(i:end,:);
    mC = mC + fK(i/iH)*(mC1+mC1');
end
end