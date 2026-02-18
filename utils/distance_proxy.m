function [ED,FN,PF,SL] = distance_proxy(data,data_in,Sigma,scale)
[T,N] = size(data); ED = zeros(T,1); FN = zeros(T,1); PF = zeros(T,1); SL = zeros(T,1); 
b=3; a=0.99;
for t = 1:T
    mY = [data_in(t+1:end,:); data(1:t,:)];
    mHt_proxy = a*(data(t,:)'*data(t,:)) + (1-a)*cov(mY);
    ED(t) = vech(mHt_proxy-Sigma(:,:,t))'*vech(mHt_proxy-Sigma(:,:,t));
    FN(t) = trace((mHt_proxy-Sigma(:,:,t))'*(mHt_proxy-Sigma(:,:,t)));
    PF(t) = trace(mHt_proxy^b-Sigma(:,:,t)^b)/(b*(b-1)) - trace(Sigma(:,:,t)^(b-1)*(mHt_proxy-Sigma(:,:,t)))/(b-1);
    SL(t) = trace(Sigma(:,:,t)\mHt_proxy)+logdet(Sigma(:,:,t))-logdet(mHt_proxy)-N;
end
ED = ED./scale^4; FN = FN./scale^4; PF = PF./(scale^(2*b));


