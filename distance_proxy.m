function [ED,FN] = distance_proxy(data,Sigma,scale)

[T,~] = size(data); ED = zeros(T,1); FN = zeros(T,1);
for t = 1:T
    ED(t) = vech((data(t,:)'*data(t,:))-Sigma(:,:,t))'*vech((data(t,:)'*data(t,:))-Sigma(:,:,t));
    FN(t) = trace(((data(t,:)'*data(t,:))-Sigma(:,:,t))'*((data(t,:)'*data(t,:))-Sigma(:,:,t)));
end
ED = ED./scale^4; FN = FN./scale^4;


