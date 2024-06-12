function [data,Sigma] = DGP(T0,N,DGP)

T = T0+3;
if (DGP==1)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%% DGP 1: DCC-type decomposition %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Define the univariate GARCH(1,1) processes
    h2 = zeros(T,N); h2(1,:) = 0.005.*ones(1,N);
    h = zeros(T,N); h(1,:) = sqrt(h2(1,:));
    % Define the variance-covariance and correlation matrices
    Sigma = zeros(N,N,T); Correlation = zeros(N,N,T);
    % Simulate the univariate GARCH(1,1) parameters (satisfying the
    % stationarity constraints)
    constant = 0.005 + (0.01-0.005)*rand(1,N);
    [a_garch,b_garch] = simulate_garch_param(N);
    % Define the T x N matrix of observations
    data = zeros(T,N);
    
    % gamma: degree of freedom of the Student distribution
    gamma = 3;
    
    % setting the trajectories for the correlations
    normalisation = [600;800;1000;1200;1400];
    a_coeff = 1+round(rand(N*(N-1)/2,1)*(length(normalisation)-1));
    b_select = 1+round(rand(N*(N-1)/2,1)*3); gg = {'cos','sin','const','mode'};
    coefficient = [0.05+(0.2-0.05)*rand(N*(N-1)/2,1),0.3+(0.6-0.3)*rand(N*(N-1)/2,1)]; 
    d_const = randi([1 T],1);
    L = zeros(N*(N-1)/2,T);
    
    for t = 2:T
        
        h2(t,:) = constant + b_garch.*h2(t-1,:) + a_garch.*(data(t-1,:).^2);
        h(t,:) = sqrt(h2(t,:));
        
        for ii = 1:N*(N-1)/2
            option = char(gg(b_select(ii)));
            switch option
                case 'cos'
                    pp = cos(2*pi*t/normalisation(a_coeff(ii)));
                case 'sin'
                    pp = sin(2*pi*t/normalisation(a_coeff(ii)));
                case 'mode'
                    pp = mode(t/normalisation(a_coeff(ii)),1);
                case 'const'
                    pp = double(t>d_const);
            end
            L(ii,t) = coefficient(ii,1)+coefficient(ii,2)*pp;
        end
        clear ii
        
        C = tril(vech_off(L(:,t),N)); Ctemp = C*C';
        Correlation(:,:,t) = Ctemp./(sqrt(diag(Ctemp))*sqrt(diag(Ctemp))');
        Sigma(:,:,t) = diag(h(t,:))*Correlation(:,:,t)*diag(h(t,:));
        % Verify whether the positive-definiteness condition is satisfied
        % If not, apply a transformation using (1-b) x St + b x Id, with S the
        % variance-covariance at time t, Id the identity matrix and b the
        % user-specified coefficient of linear combination
        if (min(eig(Sigma(:,:,t)))<eps)
            b = 0.01;
            Sigma(:,:,t) = (1-b)*Sigma(:,:,t)+b*eye(N);
        end
        nu_temp = sqrt(gamma-2)*trnd(gamma,1,N)/sqrt(gamma);
        data(t,:) = (Sigma(:,:,t)^(1/2)*nu_temp')';
        
    end
    clear t
    % Discard the first matrix of Correlation and the first line of data
    data = data(2:end,:); Sigma = Sigma(:,:,4:end);
    
elseif (DGP==2)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%% DGP 2: Diagonal BEKK dynamic %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cond1 = true;
    while cond1
        Omega = -0.05+0.1*rand(N,N); Omega = Omega*Omega'/(N);
        for ii = 1:N
            Omega(ii,ii) = 0.005+0.02*rand(1);
        end
        Omega = proj_defpos(Omega);
        quant1 = eig(Omega); cond1 = min(quant1)<0.01;
    end
    cond2 = true;
    while cond2
        B = diag(0.2+0.4*rand(N,1)); A = diag(0.05+0.15*rand(N,1));
        cond2 = max(abs(eig(A^2 + B^2)))>0.999;
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
    data = data(2:end,:); Sigma = Sigma(:,:,4:end);
    
elseif (DGP==3)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%% DGP 3: Factor GARCH dynamic %%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    cond1 = true;
    while cond1
        Omega = -0.05+0.1*rand(N,N); Omega = Omega*Omega'/(N);
        for ii = 1:N
            Omega(ii,ii) = 0.005+0.02*rand(1);
        end
        Omega = proj_defpos(Omega);
        quant1 = eig(Omega); cond1 = min(quant1)<0.01;
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
    b = (-1+2*rand(N,m)).*(rand(N,m)>0.6);
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
    data = data(2:end,:); Sigma = Sigma(:,:,4:end);
end

end
