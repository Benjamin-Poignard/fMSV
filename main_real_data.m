clear
clc

% Portfolio selection: 'MSCI' or 'SP100' 
portfolio = 'MSCI'; scale = 100;

switch portfolio
    
    case 'MSCI'
        
        % MSCI portfolio, out-of-sample period: 04/01/2016 -- 03/12/2018
        % load the MSCI country stock indices: 23 assets
        Table = readtable('MSCI.xls');
        data_MSCI = Table{1:end,[2:end]};
        % transform into log-returns
        mD = scale*(log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:)));
        N = size(mD,2);
        T_period = 3500; method_dcc = 'full';
        dates = Table.CDR_US; dates = dates(2:end);
        
    case 'SP100'
        
        % S&P 100 portfolio, out-of-sample period: 01/30/2018 -- 01/23/2020
        % load the S&P 100 stock indices: 94 assets
        % the data are under the .mat format
        % they can also be found in SP100.xls
        load data_SP.mat
        Table = readtable('SP100.xls');
        % transform into log-returns
        mD = scale*(log(data(2:end,:))-log(data(1:end-1,:)));
        N = size(mD,2);
        T_period = 1100; method_dcc = 'full';
        dates = Table.Date; dates = dates(2:end);
        
end

X_in = mD(1:T_period,:); % in-sample data
X_out = mD(T_period+1:end,:); % out-of-sample data
T_out = size(X_out,1);

% Number of lags in the first step estimation of the MSV parameters:
% user specified
p=10;

% Check the number of factors via Onatski's method
m_o =factor_selection(X_in,10); m = min(max([m_o 1]),5);

%%%%%%%%%%%%%% In-sample estimation
% level: the level up to which the estimation is performed
% level = 2: the partial correlation processes located in the 2 first
% levels of the C-vine tree are estimated; the remaining partial
% correlations are set as their sample partial correlations (computed from
% the sample correlation matrix and the underlying C-vine tree model)
mHf_dcc = dcc_for(X_in,X_out);
mHf_sbekk = sbekk_for(X_in,X_out);
mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);

[mHf_fmsv_scad_g_1,~] = fmsv_for(X_in,X_out,p,1,'scad','WLS');
[mHf_fmsv_scad_g_2,~] = fmsv_for(X_in,X_out,p,2,'scad','WLS');
[mHf_fmsv_scad_g_3,~] = fmsv_for(X_in,X_out,p,3,'scad','WLS');
[mHf_fmsv_saf_1,~] = fmsv_for(X_in,X_out,p,1,'SAF','WLS');
[mHf_fmsv_saf_2,~] = fmsv_for(X_in,X_out,p,2,'SAF','WLS');
[mHf_fmsv_saf_3,~] = fmsv_for(X_in,X_out,p,3,'SAF','WLS');

% out-of-sample portfolio returns based on GMVP
iK = 11; e = zeros(T_out,iK);
for t = 1:T_out
    e(t,1) = GMVP(mHf_dcc(:,:,t))'*X_out(t,:)';
    e(t,2) = GMVP(mHf_sbekk(:,:,t))'*X_out(t,:)';
    e(t,3) = GMVP(mHf_o_1(:,:,t))'*X_out(t,:)';
    e(t,4) = GMVP(mHf_o_2(:,:,t))'*X_out(t,:)';
    e(t,5) = GMVP(mHf_o_3(:,:,t))'*X_out(t,:)';
    e(t,6) = GMVP(mHf_fmsv_scad_g_1(:,:,t))'*X_out(t,:)';
    e(t,7) = GMVP(mHf_fmsv_scad_g_2(:,:,t))'*X_out(t,:)';
    e(t,8) = GMVP(mHf_fmsv_scad_g_3(:,:,t))'*X_out(t,:)';
    e(t,9) = GMVP(mHf_fmsv_saf_1(:,:,t))'*X_out(t,:)';
    e(t,10) = GMVP(mHf_fmsv_saf_2(:,:,t))'*X_out(t,:)';
    e(t,11) = GMVP(mHf_fmsv_saf_3(:,:,t))'*X_out(t,:)';
end

% out-of-sample average portfolio returns, standard deviations and
% information ratios
Results = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]';
E_gmvp = (e-repmat(mean(e),T_out,1)).^2;

[ED_dcc,FN_dcc] = distance_proxy(X_out,mHf_dcc,scale);
[ED_bekk,FN_bekk] = distance_proxy(X_out,mHf_sbekk,scale);
[ED_o_1,FN_o_1] = distance_proxy(X_out,mHf_o_1,scale);
[ED_o_2,FN_o_2] = distance_proxy(X_out,mHf_o_2,scale);
[ED_o_3,FN_o_3] = distance_proxy(X_out,mHf_o_3,scale);

[ED_scad_g_1,FN_scad_g_1] = distance_proxy(X_out,mHf_fmsv_scad_g_1,scale);
[ED_scad_g_2,FN_scad_g_2] = distance_proxy(X_out,mHf_fmsv_scad_g_2,scale);
[ED_scad_g_3,FN_scad_g_3] = distance_proxy(X_out,mHf_fmsv_scad_g_3,scale);

[ED_saf_1,FN_saf_1] = distance_proxy(X_out,mHf_fmsv_saf_1,scale);
[ED_saf_2,FN_saf_2] = distance_proxy(X_out,mHf_fmsv_saf_2,scale);
[ED_saf_3,FN_saf_3] = distance_proxy(X_out,mHf_fmsv_saf_3,scale);

E_ed = [ED_dcc ED_bekk ED_o_1 ED_o_2 ED_o_3 ED_scad_g_1 ED_scad_g_2...
    ED_scad_g_3 ED_saf_1 ED_saf_2 ED_saf_3];
E_ed_av = 252*mean(E_ed);

E_fn = [FN_dcc FN_bekk FN_o_1 FN_o_2 FN_o_3 FN_scad_g_1 FN_scad_g_2...
    FN_scad_g_3 FN_saf_1 FN_saf_2 FN_saf_3];
E_fn_av = 252*mean(E_fn);

% Model Confidence Test GMVP
[includedR, pvalsR_gmvp, excluded] = mcs(E_gmvp,0.1,10000,12);
excl_select_model_gmvp = [excluded ;includedR];
[excl_select_model_gmvp pvalsR_gmvp]

% Model Confidence Test Euclidean distance
[includedR, pvalsR_ed, excluded] = mcs(E_ed,0.1,10000,12);
excl_select_model_ed = [excluded ;includedR];
[excl_select_model_ed pvalsR_ed]

% Model Confidence Test Frobenius norm
[includedR, pvalsR_fn, excluded] = mcs(E_fn,0.1,10000,12);
excl_select_model_fn = [excluded ;includedR];
[excl_select_model_fn pvalsR_fn]
