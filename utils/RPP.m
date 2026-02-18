function vW = RPP(mSigma)
% weights for risk parity portfolio
iN = size(mSigma,1);
vI = ones(iN,1);

iCont=1; vW = vI./sqrt(diag(mSigma)); iC = 0.5;
while (iCont>0)
    vF = mSigma*vW - iC*(vI./vW);
    % Optimize vW wthout constraint 
    % mJ = mSigma + iC*diag(vI./(vW.^2));
    % vW_new = vW - mJ\vF;
    %
    % consider vW = e^theta and optimize theta for positivity constraint
    mJ = mSigma*diag(vW) + iC*diag(vI./vW);
    vW_new = exp(log(vW) - mJ\vF);
    iCheck = max(abs(vW_new-vW));
    vW = vW_new;
    if (iCheck<1e-05)
        iCont=0;
    end
end
vW = (1/sum(vW))*vW;





