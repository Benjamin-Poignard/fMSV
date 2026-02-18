function mA_KS = generate_SV_KS0(mX,vC,mPhi,mSig_alpha,mSig_zeta)

% Application of the Kalman filter for stationary case
[iT,iP] = size(mX);
iC1 = sum(sum(isnan(vC))) +sum(sum(isnan(mPhi)));
iC2 = sum(sum(isnan(mSig_alpha))) +sum(sum(isnan(mSig_zeta)));
if (iC1+iC2>0)
    mA_KS = -Inf*ones(iT,iP);
else
    mSig_eta = mSig_alpha;
    
    mL_v = chol(mSig_eta,'lower');
    mL_u = chol(mSig_zeta,'lower');
    mG = [mL_u zeros(iP,iP)];
    mH = [zeros(iP,iP) mL_v];
    mZ = eye(iP);
    
    mJJ = zeros(iT,2*iP*iP);
    mLL = zeros(iT,iP*iP);
    imDD = zeros(iT,0.5*iP*(iP+1));mPP = zeros(iT,0.5*iP*(iP+1));
    mA = zeros(iT,iP); mEE = zeros(iT,iP);
    vA =zeros(1,iP); 
    Pt = mSig_eta;
    for t=1:iT
        mPP(t,:) = vech(Pt)';    mA(t,:) = vA;
        vE = mX(t,:) - vC' - vA*mZ';%vE
        mD = mZ*Pt*mZ' + mG*mG';
        imD = inv(mD);
        mK = (mPhi*Pt*mZ'+mH*mG')*imD;
        mJ = mH - mK*mG;
        mL = mPhi -mK*mZ;
        vA = vA*mPhi' + vE*mK';
        Pt = mPhi*Pt*mL' + mH*mJ';
        mJJ(t,:) = vec(mJ)'; mLL(t,:) = vec(mL)';
        imDD(t,:) = vech(imD)';
        mEE(t,:) = vE;
    end
    
    mU = zeros(iP,iP); vR = zeros(1,iP);
    mA_KS = zeros(iT,iP);
    for t=iT:-1:1
        vE = mEE(t,:);
        imD = unvech(imDD(t,:)'); Pt = unvech(mPP(t,:)');
        mL = reshape(mLL(t,:),iP,iP);
        vR = vE*imD*mZ + vR*mL;
        mU = mZ'*imD*mZ + mL'*mU*mL;
        mA_KS(t,:) = mA(t,:) + vR*Pt;
    end
    mA_KS = mA_KS + vC';
end

end
