function [like,mH] = sbekk_Clikelihood(theta,mY,mS)
[iT,iM] = size(mY);
iA = theta(1);
iB = theta(2);

vN = (1-iA-iB)*vec(mS);
vH = vec(mS)';
ll=0;
mH = zeros(iM,iM,iT);
mH(:,:,1)= mS;
for t=2:iT
    vH = vN'+ iA*vec(mY(t-1,:)'*mY(t-1,:))' + iB*vH;
    mHt = reshape(vH,iM,iM);
    mH(:,:,t) = mHt;
    CL=0;
    for ii=1:iM-1
        index1 = ii; index2 = ii+1;
        mH_temp = [mHt(index1,index1),mHt(index1,index2);mHt(index2,index1),mHt(index2,index2)]; obs = [mY(t,index1),mY(t,index2)];
        CL = CL+log(det(mH_temp))+obs*inv(mH_temp)*obs';
    end
    ll = ll + CL/(iM-1);
end
like = ll/2;
end