%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for the replicaiton of the simulation experiments
% DGP 1: BEKK-based DGP with dimensions p = 20, p = 100, p = 500
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DGP 1, p = 20
addpath(genpath(pwd))
clear
clc
rng(1,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=20;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=1;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end

%% DGP 1, p = 100
addpath(genpath(pwd))
clear
clc
rng(2,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=100;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=1;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end

%% DGP 1, p = 500
addpath(genpath(pwd))
clear
clc
rng(3,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=500;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=1;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Code for the replicaiton of the simulation experiments
% DGP 2: Factor model-based DGP with dimensions p = 20, p = 100, p = 500
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DGP 2, p = 20
addpath(genpath(pwd))
clear
clc
rng(4,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=20;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=2;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end

%% DGP 2, p = 100
addpath(genpath(pwd))
clear
clc
rng(5,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=100;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=2;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end

%% DGP 2, p = 500
addpath(genpath(pwd))
clear
clc
rng(6,"twister"); cvx_setup
% Monte Carlo experiments
% Model comparison using in-sample covariance estimates
% scalar DCC, scalar BEKK, fMSV

vN=500;
T=2000;
iR=120; % number of simulations
iK=12; % number of models

N = vN; iM=2;
mRf = zeros(iR,iK); mRe = zeros(iR,iK);
mRls = zeros(iR,iK); mRlb = zeros(iR,iK);
p=10;

for oo=1:iR
    
    [X_in,Sigma_in] = DGP(T,N,iM); X_out = X_in;
    
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
    
    mRe(oo,:) = [ED_dcc ED_sbekk ED_o_1 ED_o_2 ED_o_3 ED_o_4 ED_o_5 ED_fmsv_1 ED_fmsv_2 ED_fmsv_3 ED_fmsv_4 ED_fmsv_5];
    
    mRf(oo,:) = [FN_dcc FN_sbekk FN_o_1 FN_o_2 FN_o_3 FN_o_4 FN_o_5 FN_fmsv_1 FN_fmsv_2 FN_fmsv_3 FN_fmsv_4 FN_fmsv_5];
    
    mRls(oo,:) = [LS_dcc LS_sbekk LS_o_1 LS_o_2 LS_o_3 LS_o_4 LS_o_5 LS_fmsv_1 LS_fmsv_2 LS_fmsv_3 LS_fmsv_4 LS_fmsv_5];
    
    mRlb(oo,:) = [LB_dcc LB_sbekk LB_o_1 LB_o_2 LB_o_3 LB_o_4 LB_o_5 LB_fmsv_1 LB_fmsv_2 LB_fmsv_3 LB_fmsv_4 LB_fmsv_5];
    
end