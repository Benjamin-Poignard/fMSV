function [like,mH] = sbekk_likelihood(theta,mY,mS)
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
      ll = ll + 0.5*log(det(mHt)) + 0.5*mY(t,:)*(mHt\mY(t,:)');
  end
  like = ll;
end
  
  
  