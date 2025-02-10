function [data,Sigma] = DGP(T0,N,DGP)

% Inputs: 
%         - T0: sample size
%         - N: numnber of variables
%         - DGP: ==1 : Diagonal BEKK; ==2 Factor GARCH 
% Outputs:
%         - data: T0 x N data matrix of observations 
%         - Sigma: N x N x T0 variance-covariance matrix (true) of data

T = T0+100;
if (DGP==1)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% DGP 1: Diagonal BEKK dynamic %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Omega = -0.05+0.1*rand(N,N); Omega = Omega*Omega'/(N);
    for ii = 1:N
        Omega(ii,ii) = 0.005+0.02*rand(1);
    end
    if min(eig(Omega))<0.01
        zeta = 0;
        while (min(eig(Omega))<0.01)
            Omega = Omega + (zeta + abs(min(eig(Omega))))*eye(N);
            zeta = zeta + 0.005;
        end
    end
    cond = true;
    while cond
        B = diag(0.2+0.4*rand(N,1)); A = diag(0.05+0.15*rand(N,1));
        cond = max(abs(eig(A^2 + B^2)))>0.999;
    end

    Sigma = zeros(N,N,T); data = zeros(T,N); gamma = 3;

    for t = 2:T

        Sigma(:,:,t) = Omega + A*data(t-1,:)'*data(t-1,:)*A + B*Sigma(:,:,t-1)*B;

        % Simulate the data in the Student distribution with gamma degrees
        % of freedom. Alternatively, the data can be simulated in a
        % multivariate normal distribution
        nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
        data(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';

    end
    data = data(101:end,:); Sigma = Sigma(:,:,101:end);

elseif (DGP==2)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% DGP 2: Factor GARCH dynamic %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Omega = -0.05+0.1*rand(N,N); Omega = Omega*Omega'/(N);
    for ii = 1:N
        Omega(ii,ii) = 0.005+0.02*rand(1);
    end
    if min(eig(Omega))<0.01
        zeta = 0;
        while (min(eig(Omega))<0.01)
            Omega = Omega + (zeta + abs(min(eig(Omega))))*eye(N);
            zeta = zeta + 0.005;
        end
    end

    % number of factors
    m = 2;
    % Define the univariate GARCH(1,1) processes
    h2 = zeros(T,m); h2(1,:) = 0.005.*ones(1,m);
    % Define the variance-covariance and correlation matrices
    Sigma = zeros(N,N,T);
    % Simulate the univariate GARCH(1,1) parameters (satisfying the
    % stationarity constraints)
    constant = 0.005 + (0.01-0.005)*rand(1,m);
    [a_garch,b_garch] = simulate_garch_param(m);
    data = zeros(T,N); factors = zeros(T,m); gamma = 3;

    % factor parameters assumed sparse
    b = (-0.8+1.6*rand(N,m)).*(rand(N,m)>0.6);
    for t = 2:T

        h2(t,:) = constant + b_garch.*h2(t-1,:) + a_garch.*(factors(t-1,:).^2);
        Sigma(:,:,t) = Omega;
        for k = 1:m
            Sigma(:,:,t) = Sigma(:,:,t) + sqrt(h2(t,k))*b(:,k)*b(:,k)';
            factors(t,k) = mvnrnd(0,h2(t,k));
        end
        nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
        data(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
    end
    data = data(101:end,:); Sigma = Sigma(:,:,101:end);
end

end
