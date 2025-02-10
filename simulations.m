%% DGP 1: BEKK-based process
clear
clc
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, O-GARCH, fMSV

% Dimension
N = 50; % 100, 150
% Sample size
T=2000;
% DGP: iM = 1 for BEKK-based DGP; iM = 2 for factor-based DGP
iM=1;

[X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;

% Number of lags in the first step estimation of the MSV parameters:
% user specified
p=10;

mHf_dcc = dcc_for(X_in,X_out);
mHf_sbekk = sbekk_for(X_in,X_out);

mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);

[mHf_fmsv_scad_g_1,~] = fmsv_for(X_in,X_out,p,1,'SFM','WLS');
[mHf_fmsv_scad_g_2,~] = fmsv_for(X_in,X_out,p,2,'SFM','WLS');
[mHf_fmsv_scad_g_3,~] = fmsv_for(X_in,X_out,p,3,'SFM','WLS');

[mHf_fmsv_poet_1,~] = fmsv_for(X_in,X_out,p,1,'POET','WLS');
[mHf_fmsv_poet_2,~] = fmsv_for(X_in,X_out,p,2,'POET','WLS');
[mHf_fmsv_poet_3,~] = fmsv_for(X_in,X_out,p,3,'POET','WLS');

[ED_dcc,FN_dcc] = distance(Sigma_in,mHf_dcc);
[ED_sbekk,FN_sbekk] = distance(Sigma_in,mHf_sbekk);
[ED_o_1,FN_o_1] = distance(Sigma_in,mHf_o_1);
[ED_o_2,FN_o_2] = distance(Sigma_in,mHf_o_2);
[ED_o_3,FN_o_3] = distance(Sigma_in,mHf_o_3);
[ED_fmsv_scad_g_1,FN_fmsv_scad_g_1] = distance(Sigma_in,mHf_fmsv_scad_g_1);
[ED_fmsv_scad_g_2,FN_fmsv_scad_g_2] = distance(Sigma_in,mHf_fmsv_scad_g_2);
[ED_fmsv_scad_g_3,FN_fmsv_scad_g_3] = distance(Sigma_in,mHf_fmsv_scad_g_3);
[ED_fmsv_poet_1,FN_fmsv_poet_1] = distance(Sigma_in,mHf_fmsv_poet_1);
[ED_fmsv_poet_2,FN_fmsv_poet_2] = distance(Sigma_in,mHf_fmsv_poet_2);
[ED_fmsv_poet_3,FN_fmsv_poet_3] = distance(Sigma_in,mHf_fmsv_poet_3);

% Euclidean distance
mRe = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_fmsv_scad_g_1 ED_fmsv_scad_g_2 ED_fmsv_scad_g_3...
    ED_fmsv_poet_1 ED_fmsv_poet_2 ED_fmsv_poet_3 ];
% Frobenius distance
mRf = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_fmsv_scad_g_1 FN_fmsv_scad_g_2 FN_fmsv_scad_g_3...
    FN_fmsv_poet_1 FN_fmsv_poet_2 FN_fmsv_poet_3 ];

%% DGP 2: factor GARCH-based process
clear
clc
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, O-GARCH, fMSV

% Dimension
N = 50; % 100, 159
% Sample size
T=2000;
% DGP: iM = 1 for BEKK-based DGP; iM = 2 for factor-based DGP
iM=2;

[X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;

% Number of lags in the first step estimation of the MSV parameters:
% user specified
p=10;

mHf_dcc = dcc_for(X_in,X_out);
mHf_sbekk = sbekk_for(X_in,X_out);

mHf_o_1=o_mvgarch_for(X_in,X_out,1,1,0,1);
mHf_o_2=o_mvgarch_for(X_in,X_out,2,1,0,1);
mHf_o_3=o_mvgarch_for(X_in,X_out,3,1,0,1);

[mHf_fmsv_scad_g_1,~] = fmsv_for(X_in,X_out,p,1,'SFM','WLS');
[mHf_fmsv_scad_g_2,~] = fmsv_for(X_in,X_out,p,2,'SFM','WLS');
[mHf_fmsv_scad_g_3,~] = fmsv_for(X_in,X_out,p,3,'SFM','WLS');

[mHf_fmsv_poet_1,~] = fmsv_for(X_in,X_out,p,1,'POET','WLS');
[mHf_fmsv_poet_2,~] = fmsv_for(X_in,X_out,p,2,'POET','WLS');
[mHf_fmsv_poet_3,~] = fmsv_for(X_in,X_out,p,3,'POET','WLS');

[ED_dcc,FN_dcc] = distance(Sigma_in,mHf_dcc);
[ED_sbekk,FN_sbekk] = distance(Sigma_in,mHf_sbekk);
[ED_o_1,FN_o_1] = distance(Sigma_in,mHf_o_1);
[ED_o_2,FN_o_2] = distance(Sigma_in,mHf_o_2);
[ED_o_3,FN_o_3] = distance(Sigma_in,mHf_o_3);
[ED_fmsv_scad_g_1,FN_fmsv_scad_g_1] = distance(Sigma_in,mHf_fmsv_scad_g_1);
[ED_fmsv_scad_g_2,FN_fmsv_scad_g_2] = distance(Sigma_in,mHf_fmsv_scad_g_2);
[ED_fmsv_scad_g_3,FN_fmsv_scad_g_3] = distance(Sigma_in,mHf_fmsv_scad_g_3);
[ED_fmsv_poet_1,FN_fmsv_poet_1] = distance(Sigma_in,mHf_fmsv_poet_1);
[ED_fmsv_poet_2,FN_fmsv_poet_2] = distance(Sigma_in,mHf_fmsv_poet_2);
[ED_fmsv_poet_3,FN_fmsv_poet_3] = distance(Sigma_in,mHf_fmsv_poet_3);

% Euclidean distance
mRe = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_fmsv_scad_g_1 ED_fmsv_scad_g_2 ED_fmsv_scad_g_3...
    ED_fmsv_poet_1 ED_fmsv_poet_2 ED_fmsv_poet_3 ];
%Frobenius distance
mRf = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_fmsv_scad_g_1 FN_fmsv_scad_g_2 FN_fmsv_scad_g_3...
    FN_fmsv_poet_1 FN_fmsv_poet_2 FN_fmsv_poet_3 ];
