function [ED,FN] = distance(mH0,mH1)
  [N,~,T] = size(mH0);
  vED = zeros(T,1); vFN = zeros(T,1); vSL = zeros(T,1);
  for t=1:T
    H0t = reshape(mH0(:,:,t),[N N]);
    H1t = reshape(mH1(:,:,t),[N N]);
    vED(t) = vech(H0t-H1t)'*vech(H0t-H1t);
    vFN(t) = trace((H0t-H1t)'*(H0t-H1t));
  end
  ED = mean(vED); FN = mean(vFN); 
end

