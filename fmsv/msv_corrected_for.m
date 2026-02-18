function Hf = msv_corrected_for(data_in,data_out,p)

% - data_in: T_in x N vector of in-sample observations, with T_in the
%   sample size of the in-sample period
% - data_out: T_out x N vector of out-sample observations, with T_out the
%   sample size of the out-of-sample period
% - p: number of lags for the first step

% A cross-validation procedure is performed to select the optimal tuning
% parameter lambda. One can specify a wider/smaller grid
% p is the number of lag values for filtering the residuals

% Estimation of the MSV model
% Adaptive LASSO penalised MSV
[~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate_corrected(data_in,p);
% Generate the out-of-sample forecasts of the penalized MSV
H_msv_ols_alasso = generate_SV_process_corrected(data_out,p,B,Sig_zeta,Sig_alpha,Gamma);
Hf = H_msv_ols_alasso;