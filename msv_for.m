function Hf = msv_for(Rt_in,Rt_out,Pen,p,adjust)
% A cross-validation procedure is performed to select the optimal tuning
% parameter lambda. A grid search is specified around sqrt(log(p*N^2)/T),
% which is a rate generally identified as optimal in the sparse literature
% One can specify a wider/smaller grid
% p is the number of lag values for filtering the residuals

[T_in,N]=size(Rt_in);T_out = length(Rt_out);
if (adjust==0)
    returns_in = Rt_in;
    returns_out = Rt_out;
elseif (adjust==1)
    % with adjsutment on log of zero; book of Fuller (1996)
    iC=0.02;
    x = (log(Rt_in.^2+(1e-04)) -iC*ones(T_in,N)./(Rt_in.^2+iC));
    returns_in = exp(0.5*x);
    x = (log(Rt_out.^2+(1e-04)) -iC*ones(T_out,N)./(Rt_out.^2+iC));
    returns_out = exp(0.5*x);
end

% Estimation of the MSV model
% grid: user-specified grid of values for cross-validation around sqrt(log(p*N^2)/T_in)
grid = (0.01:0.1:10); lambda = grid*sqrt(log(p*N^2)/T_in);

switch Pen
    case 'pen'
    % Adaptive LASSO penalised MSV
    [~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(returns_in,p,'no-constant','pen',lambda);
    % Generate the out-of-sample forecasts of the SCAD MSV using returns_out
    H_msv_ols_alasso = generate_SV_process(returns_out,p,B,Sig_zeta,Sig_alpha,Gamma);
    Hf = H_msv_ols_alasso;
    case 'nonpen'
    % Non-penalised MSV
    lambda = 0;
    [~,B,Sig_zeta,Sig_alpha,Gamma] = SV_process_estimate(returns_in,p,'no-constant','nonpen',lambda);
    % Generate the out-of-sample forecasts of the Non-penalised MSV using returns_out
    H_msv_ols_nonpen = generate_SV_process(returns_out,p,B,Sig_zeta,Sig_alpha,Gamma);
    Hf = H_msv_ols_nonpen;
end

end