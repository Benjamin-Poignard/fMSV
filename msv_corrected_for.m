function Hf = msv_corrected_for(data_in,data_out,Pen,p)

% - data_in: T_in x N vector of in-sample observations, with T_in the
%   sample size of the in-sample period
% - data_in: T_out x N vector of out-sample observations, with T_out the
%   sample size of the out-of-sample period
% - Pen: penalisation specification
%   ==> 'pen': adaptive LASSO
%   ==> 'nonpen': non-penalized
% - p: number of lags for the first step

% A cross-validation procedure is performed to select the optimal tuning
% parameter lambda. A grid search is specified around sqrt(log(p*N^2)/T),
% which is a rate generally identified as optimal in the sparse literature
% One can specify a wider/smaller grid
% p is the number of lag values for filtering the residuals

[T_in,N]=size(data_in); T_out = size(data_out,1);
iC=0.02;
x = (log(data_in.^2+(1e-04)) -iC*ones(T_in,N)./(data_in.^2+iC));
data_in = exp(0.5*x);
z = (log(data_out.^2+(1e-04)) -iC*ones(T_out,N)./(data_out.^2+iC));
data_out = exp(0.5*z);
% Estimation of the MSV model
% grid: user-specified grid of values for cross-validation around sqrt(log(p*N^2)/T_in)
grid = (0.01:0.1:10); lambda = grid*sqrt(log(p*N^2)/T_in);

switch Pen
    
    case 'pen'
        
        % Adaptive LASSO penalised MSV
        [~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate_corrected(data_in,p,'pen',lambda);
        % Generate the out-of-sample forecasts of the penalized MSV
        H_msv_ols_alasso = generate_SV_process_corrected(data_out,p,B,Sig_zeta,Sig_alpha,Gamma);
        Hf = H_msv_ols_alasso;
        
    case 'nonpen'
        
        % Non-penalised MSV
        lambda = 0;
        [~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate_corrected(data_in,p,'nonpen',lambda);
        % Generate the out-of-sample forecasts of the non-penalised MSV
        H_msv_ols_nonpen = generate_SV_process_corrected(data_out,p,B,Sig_zeta,Sig_alpha,Gamma);
        Hf = H_msv_ols_nonpen;
        
end