function Ht=o_mvgarch_for(data_in,data_out,numfactors,p,o,q,startingVals)
% Estimates a multivariate GARCH model using Orthogonal or Factor Garch.  Involves Principal Component
% Analysis to reduce volatility modelling to univariate garches.  See Carrol 2000 (An Inrtoduction to O-Garch)
%
% USAGE:
%   Ht = o_mvgarch(DATA,NUMFACTORS,P,O,Q);
%
% INPUTS:
%   DATA       - A T by K matrix of zero mean residuals
%   NUMFACTORS - The number of principal components to include in the MV_GARCH model
%   P          - Positive, scalar integer representing the number of symmetric innovations -OR-
%                   K by 1 vector of individual symmetric innovations order
%   O          - Non-negative, scalar integer representing the number of asymmetric lags -OR-
%                   K by 1 vector of individual asymmetric innovations order
%   Q          - Non-negative, scalar integer representing the number of conditional variance lags -OR-
%                   K by 1 vector of individual conditional variance lags
%
% OUTPUTS:
%   HT         - A [K K T] dimension matrix of conditional covariances
%
% COMMENTS:
%   Models the conditional covariance of assets using the NUMFACTORS principal components with the
%   highest contribution to total variation in the data.  The conditional covariance is
%
%   H_t = W*F_t*W' + Omega
%
%   where F_t is a diagonal matrix of the conditional factor variances, W is a K by NUMFACTORS
%   matrix of factor loadings and Omega is a diagonal matrix of (time-invariant) idiosyncratic
%   variances.  If NUMFACTORS = K when Omega = 0.
%
% See also PCA, CCC_MVGARCH, DCC, GOGARCH


% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 3    Date: 10/28/2009

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
switch nargin
    case 6
        options = [];
        startingVals = [];
    case 7
        options = [];
    case 8
        % Nothing
    otherwise
        error('5 to 7 arguments required.')
end

[t,k] = size(data_in);

if min(t,k)<2 || t<k
    error('DATA must be a T by K matrix, T>K>1');
end

%p, o, q much be non-negative scalars

% p
if length(p)==1
    if p<1 || floor(p)~=p
        error('P must be a positive integer if scalar.');
    end
    p = ones(numfactors,1) * p;
else
    if length(p)~=numfactors || min(size(p))~=1 || any(p<1) || any(floor(p)~=p)
        error('P must contain K positive integer elements if a vector.');
    end
end

% o
if length(o)==1
    if o<0 || floor(o)~=o
        error('O must be a non-neagative integer if scalar.');
    end
    o = ones(numfactors,1) * o;
else
    if length(o)~=numfactors || min(size(o))~=1 || any(o<0) || any(floor(o)~=o)
        error('O must contain K non-negative integer elements if a vector.');
    end
end

% q
if length(q)==1
    if q<0 || floor(q)~=q
        error('Q must be a non-neagative integer if scalar.');
    end
    q = ones(numfactors,1) * q;
else
    if length(q)~=numfactors || min(size(q))~=1 || any(q<0) || any(floor(q)~=q)
        error('Q must contain K non-negative integer elements if a vector.');
    end
end

if  ~isempty(startingVals)
    if size(startingVals,2)>size(startingVals,1)
        startingVals = startingVals';
    end
    if length(startingVals)<(k+sum(p)+sum(o)+sum(q))
        error('STARTINGVALS should be a K+sum(P)+sum(O)+sum(Q) by 1 vector');
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input Argument Checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[w, pc] = pca(data_in,'outer');

weights = w(1:numfactors,:);
pcs = pc(:,1:numfactors);

htMat = zeros(t,numfactors);
optimoptions.Hessian = 'bfgs';
optimoptions.MaxRLPIter = 1000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 1000;
for i=1:numfactors
    [univariate{i}.parameters, univariate{i}.likelihood, univariate{i}.stderrors, univariate{i}.robustSE, univariate{i}.ht, univariate{i}.scores] ...
        = fattailed_garch(pcs(:,i),1,1,'NORMAL',[],optimoptions);
    htMat(:,i) = univariate{i}.ht;
end

if numfactors<k
    errors = data_in - pcs * weights;
    omega = diag(mean(errors.^2));
else
    omega = zeros(k);
end

ht = zeros(k,k,t);
for i=1:t
    ht(:,:,i) = weights' * diag(htMat(i,:)) * weights + omega;
end

t_out = size(data_out,1);
htMat_oos = zeros(t_out,numfactors); 

% Compute the eigenvalues and eigenvectors
[eigenvects,eigenvals]=eig(data_in'*data_in/t);
% Ensures they are actually sorted smallest to largest
[~,order] = sort(diag(eigenvals));
% Reorder the eigenvectors in the same order
eigenvects = eigenvects(:,order);
% Principle components are the data times the eigenvectors
pc_out=data_out*eigenvects; pc_out=fliplr(pc_out); pcs_out = pc_out(:,1:numfactors);
for jj=1:numfactors
    [~, htMat_oos(:,jj)] = dcc_univariate_simulate(univariate{jj}.parameters,1,1,pcs_out(:,jj));
end
clear jj
Ht = zeros(k,k,t_out);
for i=1:t_out
    Ht(:,:,i) = weights' * diag(htMat_oos(i,:)) * weights + omega;
end