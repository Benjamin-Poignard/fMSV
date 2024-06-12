function L = Loss_penalized_objective(Y,X,param,lambda,method)

% Inputs:
%         - Y: vector of response variables
%         - X: matrix of lagged variables
%         - param: vector of parameters of interest
%         - lambda: tuning parameter (user specified)
%         - method: 'scad' or 'mcp'

% Output:
%         - L: value of the objective function (OLS function)

switch method
    case 'scad'
        % a_scad = 3.7; the user may set another value
        pen = scad(param,lambda,3.7);
    case 'mcp'
        % b_mcp = 3.5; the user may set another value
        pen = mcp(param,lambda,3.5);
end
L = sum((Y-X*param).^2)/(2*length(Y)) + pen;

