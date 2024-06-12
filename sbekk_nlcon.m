function [c,ceq] = sbekk_nlcon(theta)
%  OUTPUTS:
%    C    - Vector of nonlinear inequality constraints.  Based on the roots
%           of a polynomial in beta
%    CEQ  - Empty matrix
iA = theta(1);
iB = theta(2);
iS = 1-(iA+iB);
iC = 0.99995; lb = (1-iC)*ones(2,1); ub =  iC*ones(2,1);
c = [-iS;theta-ub;lb-theta];
ceq=[];
end