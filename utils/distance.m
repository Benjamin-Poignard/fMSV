function [ED,FN,vLS,vLB,vIF] = distance(mH0,mH1)

% Squared Euclidean norm, Frobenius norm, Stein loss, D_b loss, averaged
% over the sample size, between the sequence of square matrices mH0 and mH1
[N,~,T] = size(mH0);
vED = zeros(T,1); vFN = zeros(T,1); vLS = zeros(T,1); vLB = zeros(T,1); vIF = zeros(T,1);
b = 3;
for t=1:T
    H0t = reshape(mH0(:,:,t),[N N]);
    H1t = reshape(mH1(:,:,t),[N N]);
    vED(t) = vech(H0t-H1t)'*vech(H0t-H1t);
    vFN(t) = trace((H0t-H1t)'*(H0t-H1t));
    vLS(t) = trace(H1t\H0t)-log(det(H1t\H0t))-N;
    vLB(t) = trace(H0t^b-H1t^b)/(b*(b-1)) - trace((H1t^(b-1))*(H0t-H1t))/(b-1);
end
ED = mean(vED); FN = mean(vFN); vLS = mean(vLS); vLB = mean(vLB);
end

