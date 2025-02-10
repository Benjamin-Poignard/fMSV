# fMSV

Matlab implementation of factor Multivariate Stochastic Volatility (fMSV) model based on the paper:

*Factor multivariate stochastic volatility models of high dimension* by Benjamin Poignard and Manabu Asai.

Link: https://arxiv.org/abs/2406.19033

# Overview

The code in this replication includes:

- The different DGP processes considered in the simulated experiments: the replicator should execute program *simulations.m*.
- The real data experiment for the MSCI and S&P 100 portfolios: the replicator should execute program *main_real_data.m*.

# Data availability

The MSCI and S&P 100 data used to support the findings of this study are publicly available. The MSCI data were collected from the link: https://www.msci.com. The S&P 100 data were collected from the link https://finance.yahoo.com. The S&P 500 data were downloaded (license required) from the link: https://macrobond.com.

The full sample period of the MSCI data is: 12/31/1998 - 03/12/2018. The full sample period of the S&P 100 data is: 18/02/2010 - 01/23/2020. The S&P 100 indices contains the 94 assets considered in the paper: AbbVie Inc., Dow Inc., General Motors, Kraft Heinz, Kinder Morgan and PayPal Holdings are excluded from the original dataset. 

The raw data file for the MSCI indices is *MSCI.xls*. The raw data file for the S&P 100 indices is *SP100.xls* and the S&P 100 data used in the paper (excluding AbbVie Inc., Dow Inc., General Motors, Kraft Heinz, Kinder Morgan and PayPal Holdings) are stored in *data_SP.mat*. The replicator can access both MSCI and S&P 100 datasets.

# Software requirements

The Matlab code was run on a Mac-OS Apple M1 Ultra with 20 cores and 128 GB Memory. The version of the Matlab software on which the code was run is a follows: 9.12.0.1975300 (R2022a) Update 3.

The following toolboxes should be installed:

- Statistics and Machine Learning Toolbox, Version 12.3.
- Parallel Computing Toolbox, Version 7.6.

The Parallel Computing Toolbox is highly recommended to run the code to speed up the cross-validation procedure employed to select the optimal tuning parameter. 

# Description of the code

The main function to conduct the fMSV model is *fmsv_for.m* and relies on the factor model estimation based on the Principal Orthogonal complEment Thresholding (POET) model of Fan, Liao and Mincheva (2013) and the Sparse Factor Model (SFM) model of Poignard and Terada (2025).
The repository for SFM estimation can be accessed here: https://github.com/Benjamin-Poignard/sparse-factor-models


The codes for estimating the scalar DCC model and scalar BEKK (with composited likelihood method) are provided in the replication package: the replicator should refer to dcc_mvgarch_for.m. To be precise, both full likelihood and composite-likelihood methods are implemented in the second-step objective function. The latter method is based on contiguous overlapping pairs, which builds upon C. Pakel, N. Shephard, K. Sheppard and R.F. Engle (2021) and should be used when the dimension is large (i.e., larger than 200, 300, 400). The DCC-GARCH code builds upon the MFE toolbox of K. Sheppard, https://www.kevinsheppard.com/code/matlab/mfe-toolbox/




