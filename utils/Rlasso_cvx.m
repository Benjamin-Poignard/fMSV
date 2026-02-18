function b = Rlasso_cvx(X,Y,lambda)

% CVX optimization for adaptive LASSO

% inputs: - X: T x N data matrix
%         - Y: T x 1 vector of responses
%         - lambda: regularization parameter

% output: - b: estimated parameter

[T,N] = size(X); b = zeros(N,1); epsilon = 0.000001;
XTX = X'*X; XTY = X'*Y; b_ols = XTX\XTY; weight = abs(b_ols).^(-1); 
cvx_begin quiet
    variable b(N)
    minimize( sum_square(Y-X*b)/T + lambda*sum(weight.*abs(b)))
    subject to
        sum(abs(b)) <= 1-epsilon;
cvx_end
b(abs(b)<0.00001)=0;