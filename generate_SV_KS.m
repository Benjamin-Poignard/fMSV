function mA_KS = generate_SV_KS(mX,vC,mPhi0,mSig_alpha,mSig_zeta)

% Application of the Kalman filter for non-stationary case
[iT,iP] = size(mX);
iC1 = sum(sum(isnan(vC))) +sum(sum(isnan(mPhi0)));
iC2 = sum(sum(isnan(mSig_alpha))) +sum(sum(isnan(mSig_zeta)));
if (iC1+iC2>0)
    mA_KS = -Inf*ones(iT,iP);
else
    [V,D] = eig(mPhi0);
    vE = diag(D); vE = (abs(vE)<1).*vE + (abs(vE)>1);
    mPhi = V*diag(vE)*inv(V); %modify nonstationary to unitroot process
    mSig_eta = (2/(iT+1))*mSig_alpha; % for unitroot process
    
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
        if (min(eig(mD))<1e-14)||(sum(sum(isnan(mD))))
            mD = t*mSig_eta + mSig_zeta;
        end
        imD = inv(mD);
        mK = (mPhi*Pt*mZ'+mH*mG')*imD;
        if (sum(sum(isinf(mK)+isnan(mK)))>0)
            Pt = (t-1)*mSig_eta;
            mK = (mPhi*Pt*mZ'+mH*mG')*imD;
        end
        mJ = mH - mK*mG;
        mL = mPhi -mK*mZ;
        vA = vA*mPhi' + vE*mK';
        Pt = mPhi*Pt*mL' + mH*mJ';
        if (min(eig(Pt))<1e-14)||(sum(sum(isnan(Pt))))
            Pt = t*mSig_eta;
        end
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
end

end
