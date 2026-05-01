function H = generate_SV_process_corrected(data,p,B_hat,Sig_zeta,Sig_alpha,Gamma)
% This function generates the variance covariance MSV process H from the
% parameters estimated with SV_process_estim_memo.m
% H can be both the out-of-sample variance covariance when data are the
% out-of-sample observations, or the in-sample variance covariance

% inputs: - data: T x N vector of observations
%         - p: number of lags for the first step
%         - B_hat, Sig_zeta, Sig_alpha, Gamma are all the model parameters
%           obtained with SV_process_estim_memo.m

% output: - H: N x N x T variance covariance process


% T: number of simulated observations; N: dimension of the vector
[T,N] = size(data);

%%%%%% define the vectors and matrix %%%%%%
% Sigma is the true variance covariance matrix
% x corresponds to the log(data^2)
% x = log(data.^2)';
iC = 1e-4; v = mean(data.^2);
x = log(data.^2+ iC*v)' - (iC*(ones(T,1)*v)./(data.^2+ iC*v))';

c_star = B_hat(:,1); B_hat(:,1) = [];
Phi_sec = B_hat(:,1:N); B_hat(:,1:N) = [];

% Compute the following quantities
% c_hat = inv(eye(N)-Phi_sec)*c_star;
c_hat = (eye(N)-Phi_sec)\c_star;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fourth step: MMSLE %%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

c_vec = kron(ones(T,1),c_hat); % size of the vector: TN

if (max(abs(eig(Phi_sec)))<1)
  %  % stationary case
  %  % computation of the V_alpha matrix
  %  V_alpha = fV(Sig_alpha,Phi_sec,T);
  %  V_zeta = kron(eye(T),Sig_zeta);
  %  
  %  % Obtain V_x
  %  V_x = V_alpha+V_zeta;
  %  
  %  % Compute x_tilde
  %  %x_tilde = V_alpha*inv(V_x)*(vec(x)-c_vec)+c_vec;
  %  x_tilde = V_alpha*((V_x)\(vec(x)-c_vec))+c_vec;
  %  
  %  % Generate the \tilde{D}_t matrix
  %  xx_tilde = reshape(x_tilde,N,T);
  xx_tilde = generate_SV_KS0(x',c_hat,Phi_sec,Sig_alpha,Sig_zeta)';
else
    % apply Kalman filter and smoother for non-stationary case
    mA_KS = generate_SV_KS(x',c_hat,Phi_sec,Sig_alpha,Sig_zeta);
    xx_tilde = mA_KS';
end

d_bar = zeros(N,1); d_tilde = zeros(T,N); %data_m = mean(data);
for ii = 1:N
    d_bar(ii) = sqrt(sum((data(:,ii).^2).*exp(-xx_tilde(ii,:)'))/(T-p));
    %d_bar(ii) = sqrt(sum(((data(:,ii)-data_m(ii)).^2).*exp(-xx_tilde(ii,:)'))/(T-p));
    d_tilde(:,ii) = d_bar(ii)*exp(0.5*xx_tilde(ii,:)');
end

% Generate the variance covariance matrix H using \tilde{D}_t and Gamma
H = zeros(N,N,T);
for t = 1:T
    H(:,:,t) = diag(d_tilde(t,:))*Gamma*diag(d_tilde(t,:));
    if (sum(sum(isnan(H(:,:,t)))))||(sum(sum(isinf(H(:,:,t)))))
        H(:,:,t)=0;
    end
end
