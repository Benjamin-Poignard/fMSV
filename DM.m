function [alpha,se_a,t_value] = DM(vE)
  iF = length(vE);
  alpha = mean(vE);
  iV = HAC(vE-alpha,ones(iF,1))/(iF);
  se_a = sqrt(iV);
  t_value = alpha/se_a;
  %[alpha se_a t_value]
end