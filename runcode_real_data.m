% Real data experiment
addpath(genpath(pwd))
clear
clc
% Portfolio selection: 'MSCI' or 'SP100'
portfolio = 'MSCI'; scale = 100;
cvx_setup

switch portfolio

    case 'MSCI'

        % MSCI portfolio, out-of-sample period: 04/01/2016 -- 03/12/2018
        % load the MSCI country stock indices: 23 assets
        Table = readtable('MSCI.xls');
        data_MSCI = Table{1:end,[2:end]};
        % transform into log-returns
        mD = scale*(log(data_MSCI(2:end,:))-log(data_MSCI(1:end-1,:)));
        N = size(mD,2);
        T_period = 3900; method_dcc = 'full';
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% In-sample estimation - Out-of-sample prediction
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mHf_dcc = dcc_for(X_in,X_out);
mHf_sbekk = sbekk_for(X_in,X_out);
mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);
mHf_o_4=o_mvgarch_for(X_in,X_out,4,1,0,1);
mHf_o_5=o_mvgarch_for(X_in,X_out,5,1,0,1);

[mHf_fmsv_1,~] = fmsv_for(X_in,X_out,p,1,'WLS');
[mHf_fmsv_2,~] = fmsv_for(X_in,X_out,p,2,'WLS');
[mHf_fmsv_3,~] = fmsv_for(X_in,X_out,p,3,'WLS');
[mHf_fmsv_4,~] = fmsv_for(X_in,X_out,p,4,'WLS');
[mHf_fmsv_5,~] = fmsv_for(X_in,X_out,p,5,'WLS');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% out-of-sample portfolio returns based on GMVP - RPP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iK = 16; e = zeros(T_out,iK); v = zeros(T_out,iK);
for t = 1:T_out
    e(t,1) = GMVP(mHf_dcc(:,:,t))'*X_out(t,:)';
    e(t,2) = GMVP(mHf_sbekk(:,:,t))'*X_out(t,:)';
    e(t,3) = GMVP(mHf_o_1(:,:,t))'*X_out(t,:)';
    e(t,4) = GMVP(mHf_o_2(:,:,t))'*X_out(t,:)';
    e(t,5) = GMVP(mHf_o_3(:,:,t))'*X_out(t,:)';
    e(t,6) = GMVP(mHf_o_4(:,:,t))'*X_out(t,:)';
    e(t,7) = GMVP(mHf_o_5(:,:,t))'*X_out(t,:)';
    e(t,8) = GMVP(mHf_fmsv_1(:,:,t))'*X_out(t,:)';
    e(t,9) = GMVP(mHf_fmsv_2(:,:,t))'*X_out(t,:)';
    e(t,10) = GMVP(mHf_fmsv_3(:,:,t))'*X_out(t,:)';
    e(t,11) = GMVP(mHf_fmsv_4(:,:,t))'*X_out(t,:)';
    e(t,12) = GMVP(mHf_fmsv_5(:,:,t))'*X_out(t,:)';
    e(t,13) = (1/N)*ones(1,N)*X_out(t,:)';
    e(t,14) = GMVP(cov(X_in))'*X_out(t,:)';
    e(t,15) = GMVP(GIS(X_in))'*X_out(t,:)';
    e(t,16) = GMVP(cov1Para(X_in))'*X_out(t,:)';

    v(t,1) = RPP(mHf_dcc(:,:,t))'*X_out(t,:)';
    v(t,2) = RPP(mHf_sbekk(:,:,t))'*X_out(t,:)';
    v(t,3) = RPP(mHf_o_1(:,:,t))'*X_out(t,:)';
    v(t,4) = RPP(mHf_o_2(:,:,t))'*X_out(t,:)';
    v(t,5) = RPP(mHf_o_3(:,:,t))'*X_out(t,:)';
    v(t,6) = RPP(mHf_o_4(:,:,t))'*X_out(t,:)';
    v(t,7) = RPP(mHf_o_5(:,:,t))'*X_out(t,:)';
    v(t,8) = RPP(mHf_fmsv_1(:,:,t))'*X_out(t,:)';
    v(t,9) = RPP(mHf_fmsv_2(:,:,t))'*X_out(t,:)';
    v(t,10) = RPP(mHf_fmsv_3(:,:,t))'*X_out(t,:)';
    v(t,11) = RPP(mHf_fmsv_4(:,:,t))'*X_out(t,:)';
    v(t,12) = RPP(mHf_fmsv_5(:,:,t))'*X_out(t,:)';
    v(t,13) = (1/N)*ones(1,N)*X_out(t,:)';
    v(t,14) = RPP(cov(X_in))'*X_out(t,:)';
    v(t,15) = RPP(GIS(X_in))'*X_out(t,:)';
    v(t,16) = RPP(cov1Para(X_in))'*X_out(t,:)';
end

% out-of-sample average portfolio returns, standard deviations and
% information ratios
% The rows in Results are:
% scalar DCC - scalar BEKK - factor GARCH(m=1) - factor GARCH(m=2) - factor GARCH(m=3) - factor GARCH(m=4) - factor GARCH(m=5)
% - factor MSV(m=1) - factor MSV(m=2) - factor MSV(m=3) - factor MSV(m=4) - factor MSV(m=5)
% - equally weighed - SCov - GIS - Cov1Para
Results = [252*mean(e);sqrt(252)*std(e);(252*mean(e))./(sqrt(252)*std(e))]'
E_gmvp = (e-repmat(mean(e),T_out,1)).^2;

%V_Results = [252*mean(v);sqrt(252)*std(v);(252*mean(v))./(sqrt(252)*std(v))]'
%V_rpp =  0.5*((v-repmat(mean(v),T_out,1)).^2) -v;
z_rpp = sqrt(252)*v./(ones(T_out,1)*std(v));
V_rpp = max(max(z_rpp)) - z_rpp;
V_Results = [252*mean(v);sqrt(252)*std(v);mean(z_rpp)]'

[ED_dcc,FN_dcc,PF_dcc,SL_dcc] = distance_proxy(X_out,X_in,mHf_dcc,scale);
[ED_bekk,FN_bekk,PF_bekk,SL_bekk] = distance_proxy(X_out,X_in,mHf_sbekk,scale);
[ED_o_1,FN_o_1,PF_o_1,SL_o_1] = distance_proxy(X_out,X_in,mHf_o_1,scale);
[ED_o_2,FN_o_2,PF_o_2,SL_o_2] = distance_proxy(X_out,X_in,mHf_o_2,scale);
[ED_o_3,FN_o_3,PF_o_3,SL_o_3] = distance_proxy(X_out,X_in,mHf_o_3,scale);
[ED_o_4,FN_o_4,PF_o_4,SL_o_4] = distance_proxy(X_out,X_in,mHf_o_4,scale);
[ED_o_5,FN_o_5,PF_o_5,SL_o_5] = distance_proxy(X_out,X_in,mHf_o_5,scale);

[ED_fmsv_1,FN_fmsv_1,PF_fmsv_1,SL_fmsv_1] = distance_proxy(X_out,X_in,mHf_fmsv_1,scale);
[ED_fmsv_2,FN_fmsv_2,PF_fmsv_2,SL_fmsv_2] = distance_proxy(X_out,X_in,mHf_fmsv_2,scale);
[ED_fmsv_3,FN_fmsv_3,PF_fmsv_3,SL_fmsv_3] = distance_proxy(X_out,X_in,mHf_fmsv_3,scale);
[ED_fmsv_4,FN_fmsv_4,PF_fmsv_4,SL_fmsv_4] = distance_proxy(X_out,X_in,mHf_fmsv_4,scale);
[ED_fmsv_5,FN_fmsv_5,PF_fmsv_5,SL_fmsv_5] = distance_proxy(X_out,X_in,mHf_fmsv_5,scale);


% Model indexing for MCS performances Euclidean/Frobenius/Stein/Db distances: 
% 1. scalar DCC 
% 2. scalar BEKK
% 3. factor GARCH(m=1)
% 4. factor GARCH(m=2)
% 5. factor GARCH(m=3)
% 6. factor GARCH(m=4)
% 7. factor GARCH(m=5)
% 8. factor MSV(m=1)
% 9. factor MSV(m=2)
% 10. factor MSV(m=3)
% 11. factor MSV(m=4)
% 12. factor MSV(m=5)

% Model indexing for MCS performances GMVP/RPP: 
% 1. scalar DCC 
% 2. scalar BEKK
% 3. factor GARCH(m=1)
% 4. factor GARCH(m=2)
% 5. factor GARCH(m=3)
% 6. factor GARCH(m=4)
% 7. factor GARCH(m=5)
% 8. factor MSV(m=1)
% 9. factor MSV(m=2)
% 10. factor MSV(m=3)
% 11. factor MSV(m=4)
% 12. factor MSV(m=5)
% 13. 1/p equally weighted
% 14. SCov
% 15. GIS
% 16. Cov1Para

E_ed = [ED_dcc ED_bekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5...
        ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
E_ed_av = 252*mean(E_ed);

E_fn = [FN_dcc FN_bekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5...
        FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
E_fn_av = 252*mean(E_fn);

E_pf = [PF_dcc PF_bekk PF_o_1 PF_o_2 PF_o_3 PF_o_4 PF_o_5...
        PF_fmsv_1 PF_fmsv_2 PF_fmsv_3 PF_fmsv_4 PF_fmsv_5];
E_pf_av = 252*mean(E_pf);

E_sl = [SL_dcc SL_bekk SL_o_1 SL_o_2 SL_o_3 SL_o_4 SL_o_5...
        SL_fmsv_1 SL_fmsv_2 SL_fmsv_3 SL_fmsv_4 SL_fmsv_5];
E_sl_av = 252*mean(E_sl);
Distance = [E_ed_av' E_fn_av' E_sl_av' E_pf_av']
format bank
DistanceConst = [100*E_ed_av' 100*E_fn_av' (1e-04)*E_sl_av' (1e+03)*E_pf_av']
format short

'--------------- MCS for GMVP and RPP --------------'

% Model Confidence Test GMVP
[includedR, pvalsR_gmvp, excluded] = mcs(E_gmvp,0.1,10000,12);
excl_select_model_gmvp = [excluded ;includedR];
[excl_select_model_gmvp pvalsR_gmvp]

% Model Confidence Test RPP
[includedR, pvalsR_rpp, excluded] = mcs(V_rpp,0.1,10000,12);
excl_select_model_rpp = [excluded ;includedR];
[excl_select_model_rpp pvalsR_rpp]

'--------------- MCS for Distance Measures --------------'

% Model Confidence Test Euclidean distance
[includedR, pvalsR_ed, excluded] = mcs(E_ed,0.1,10000,12);
excl_select_model_ed = [excluded ;includedR];
[excl_select_model_ed pvalsR_ed]

% Model Confidence Test Frobenius norm
[includedR, pvalsR_fn, excluded] = mcs(E_fn,0.1,10000,12);
excl_select_model_fn = [excluded ;includedR];
[excl_select_model_fn pvalsR_fn]

% Model Confidence Test Stein Loss
[includedR, pvalsR_sl, excluded] = mcs(E_sl,0.1,10000,12);
excl_select_model_sl = [excluded ;includedR];
[excl_select_model_sl pvalsR_sl]

% Model Confidence Test Laurent et al. (2012)
[includedR, pvalsR_pf, excluded] = mcs(E_fn,0.1,10000,12);
excl_select_model_pf = [excluded ;includedR];
[excl_select_model_pf pvalsR_pf]

