%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for the replicaiton of the simulation experiments
% DGP 1: BEKK-based DGP with dimensions p = 20, p = 100, p = 500
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DGP 1
addpath(genpath(pwd))
clear
clc
% setting up cvx
cvx_setup

% Select dimension: p = 20; p = 100; p = 500;
p=100;
% Sample size
T=2000;

% DGP: iM = 1 for BEKK-based DGP
iM=1; 
% q: number of lags for MSV parameter estimation
q=10;
% data generation
[X_in,Sigma_in] = DGP(T,p,iM); X_out = X_in;

% scalar DCC
mHf_dcc = dcc_for(X_in,X_out);
% scalar BEKK
mHf_sbekk = sbekk_for(X_in,X_out);

% factor GARCH model with m = 1, 2, 3, 4, 5 factors
mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);
mHf_o_4=o_mvgarch_for(X_in,X_out,4,1,0,1);
mHf_o_5=o_mvgarch_for(X_in,X_out,5,1,0,1);

% factor MSV model with m = 1, 2, 3, 4, 5 factors
[mHf_fmsv_1,~] = fmsv_for(X_in,X_out,q,1,'WLS');
[mHf_fmsv_2,~] = fmsv_for(X_in,X_out,q,2,'WLS');
[mHf_fmsv_3,~] = fmsv_for(X_in,X_out,q,3,'WLS');
[mHf_fmsv_4,~] = fmsv_for(X_in,X_out,q,4,'WLS');
[mHf_fmsv_5,~] = fmsv_for(X_in,X_out,q,5,'WLS');

% compute the matrix distances, averaged over the sample period
[ED_dcc,FN_dcc,LS_dcc,LB_dcc] = distance(Sigma_in,mHf_dcc);
[ED_sbekk,FN_sbekk,LS_sbekk,LB_sbekk] = distance(Sigma_in,mHf_sbekk);

[ED_o_1,FN_o_1,LS_o_1,LB_o_1] = distance(Sigma_in,mHf_o_1);
[ED_o_2,FN_o_2,LS_o_2,LB_o_2] = distance(Sigma_in,mHf_o_2);
[ED_o_3,FN_o_3,LS_o_3,LB_o_3] = distance(Sigma_in,mHf_o_3);
[ED_o_4,FN_o_4,LS_o_4,LB_o_4] = distance(Sigma_in,mHf_o_4);
[ED_o_5,FN_o_5,LS_o_5,LB_o_5] = distance(Sigma_in,mHf_o_5);

[ED_fmsv_1,FN_fmsv_1,LS_fmsv_1,LB_fmsv_1] = distance(Sigma_in,mHf_fmsv_1);
[ED_fmsv_2,FN_fmsv_2,LS_fmsv_2,LB_fmsv_2] = distance(Sigma_in,mHf_fmsv_2);
[ED_fmsv_3,FN_fmsv_3,LS_fmsv_3,LB_fmsv_3] = distance(Sigma_in,mHf_fmsv_3);
[ED_fmsv_4,FN_fmsv_4,LS_fmsv_4,LB_fmsv_4] = distance(Sigma_in,mHf_fmsv_4);
[ED_fmsv_5,FN_fmsv_5,LS_fmsv_5,LB_fmsv_5] = distance(Sigma_in,mHf_fmsv_5);

% Matrix distance computation (the lower, the better)
% mRe: Euclidean distance
mRe = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
% mRf: squared Frobenius norm
mRf = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
% mRls: Stein loss
mRls = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
% mRlb: asymmetric Db distance
mRlb = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];

% The columns in each distance are:
% scalar DCC - scalar BEKK - factor GARCH(m=1) - factor GARCH(m=2) - factor GARCH(m=3) - factor GARCH(m=4) - factor GARCH(m=5)
% - factor MSV(m=1) - factor MSV(m=2) - factor MSV(m=3) - factor MSV(m=4) - factor MSV(m=5) 

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for the replicaiton of the simulation experiments
% DGP 2: Factor model-based DGP with dimensions p = 20, p = 100, p = 500
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DGP 2
addpath(genpath(pwd))
clear
clc
cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

% Select dimension: p = 20; p = 100; p = 500;
p=100;
% Sample size
T=2000;

% DGP: iM = 1 for BEKK-based DGP
iM=2; 
% q: number of lags for MSV parameter estimation
q=10;
% data generation
[X_in,Sigma_in] = DGP(T,p,iM); X_out = X_in;

% scalar DCC
mHf_dcc = dcc_for(X_in,X_out);
% scalar BEKK
mHf_sbekk = sbekk_for(X_in,X_out);

% factor GARCH model with m = 1, 2, 3, 4, 5 factors
mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);
mHf_o_4=o_mvgarch_for(X_in,X_out,4,1,0,1);
mHf_o_5=o_mvgarch_for(X_in,X_out,5,1,0,1);

% factor MSV model with m = 1, 2, 3, 4, 5 factors
[mHf_fmsv_1,~] = fmsv_for(X_in,X_out,q,1,'WLS');
[mHf_fmsv_2,~] = fmsv_for(X_in,X_out,q,2,'WLS');
[mHf_fmsv_3,~] = fmsv_for(X_in,X_out,q,3,'WLS');
[mHf_fmsv_4,~] = fmsv_for(X_in,X_out,q,4,'WLS');
[mHf_fmsv_5,~] = fmsv_for(X_in,X_out,q,5,'WLS');

% compute the matrix distances, averaged over the sample period
[ED_dcc,FN_dcc,LS_dcc,LB_dcc] = distance(Sigma_in,mHf_dcc);
[ED_sbekk,FN_sbekk,LS_sbekk,LB_sbekk] = distance(Sigma_in,mHf_sbekk);

[ED_o_1,FN_o_1,LS_o_1,LB_o_1] = distance(Sigma_in,mHf_o_1);
[ED_o_2,FN_o_2,LS_o_2,LB_o_2] = distance(Sigma_in,mHf_o_2);
[ED_o_3,FN_o_3,LS_o_3,LB_o_3] = distance(Sigma_in,mHf_o_3);
[ED_o_4,FN_o_4,LS_o_4,LB_o_4] = distance(Sigma_in,mHf_o_4);
[ED_o_5,FN_o_5,LS_o_5,LB_o_5] = distance(Sigma_in,mHf_o_5);

[ED_fmsv_1,FN_fmsv_1,LS_fmsv_1,LB_fmsv_1] = distance(Sigma_in,mHf_fmsv_1);
[ED_fmsv_2,FN_fmsv_2,LS_fmsv_2,LB_fmsv_2] = distance(Sigma_in,mHf_fmsv_2);
[ED_fmsv_3,FN_fmsv_3,LS_fmsv_3,LB_fmsv_3] = distance(Sigma_in,mHf_fmsv_3);
[ED_fmsv_4,FN_fmsv_4,LS_fmsv_4,LB_fmsv_4] = distance(Sigma_in,mHf_fmsv_4);
[ED_fmsv_5,FN_fmsv_5,LS_fmsv_5,LB_fmsv_5] = distance(Sigma_in,mHf_fmsv_5);

% Matrix distance computation (the lower, the better)
% mRe: Euclidean distance
mRe = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
% mRf: squared Frobenius norm
mRf = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
% mRls: Stein loss
mRls = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
% mRlb: asymmetric Db distance
mRlb = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];

% The entries in each distance are:
% scalar DCC - scalar BEKK - factor GARCH(m=1) - factor GARCH(m=2) - factor GARCH(m=3) - factor GARCH(m=4) - factor GARCH(m=5)
% - factor MSV(m=1) - factor MSV(m=2) - factor MSV(m=3) - factor MSV(m=4) - factor MSV(m=5) 
