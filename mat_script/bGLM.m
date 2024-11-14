% bGLM
%%% behavioral GLM demo
%% define variables
lt = 10000; % length of simulation / data
nB = 4;  % number of basis function for the kernel
[cosBasis, tgrid, basisPeaks] = makeRaisedCosBasis(nB, [0, 10], 1.3); % basis function
%%% true params
beta = 1; % nonlinear parameter
alpha_h = [-4:-1]*1;  % history kernel coefficient
alpha_s = [1,4,-2,-1]*-1;  % stimulus kernel coefficient
base = 1.;  %baseline

%% generate data
K_h = fliplr(alpha_h*cosBasis');  % history kernel
K_s = fliplr(alpha_s*cosBasis');  % stimulus kernel
lc = 1;  %length of smoothing
stim = conv(randn(1,lt),ones(1,lc),'same')/lc*1.;  % stimulus vector
dth = zeros(1,lt);
F = dth*0;
pad = length(K_s);
for tt=pad+1:lt
    F(tt) = stim(tt-pad+1:tt)*K_s' + dth(tt-pad:tt-1)*K_h';  % linear filtering
    dth(tt) = choice(NL(F(tt)+base,beta));  % nonlinearity and binary choice
end

%% MLE inference
algorithms = {'interior-point', 'sqp', 'sqp-legacy', 'active-set', 'trust-region-reflective'};
lfun = @(x)nll(x, dth, stim, cosBasis);  % objective function
opts = optimset('display','iter','Algorithm', algorithms{1});
opts = optimset('display','iter');
num_par = 9;
LB = [ones(1,num_par)]*-2;
UB = [ones(1,num_par)]*2;
% prs0 = [alpha_h,alpha_s,base];%
prs0 = randn(1,num_par);
% [x,fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);  % constrained optimization
[x,FVAL,EXITFLAG,OUTPUT,GRAD,HESSIAN] = fminunc(lfun, prs0, opts);

%% evaluation
beta_rec = 1;%x(1);
base_rec = x(end);
K_h_rec = fliplr(x(1:4)*cosBasis')*beta_rec;
K_s_rec = fliplr(x(5:8)*cosBasis')*beta_rec;
figure()
subplot(121)
plot(K_h_rec); hold on; plot(K_h,'--');
subplot(122)
plot(K_s_rec); hold on; plot(K_s,'--');

%% functions
%%% nonlinear function
function [P] = NL(F,beta)
    P = 1./(1+exp(-beta*F));
%     P = exp(-beta*F);
end

%%% stochastic choice
function [b] = choice(P)
    pp = rand();
    if pp<P
        b = 1;
    else
        b = 0;
    end
%     b = poissrnd(P);
end

%%% objective: negative Log-likelihood=
function [NLL] = nll(THETA, dth, stim, cosBasis)
    
    %%% params
    beta = 1;%THETA(1);
    alpha_h = THETA(1:4);
    alpha_s = THETA(5:8);
    base = THETA(9);
    
    %%% kernels
    K_h = (alpha_h*cosBasis');
    K_s = (alpha_s*cosBasis');
    
    %%% account for time
    stim_ = stim(2:end);
    dth_ = dth(2:end);
    dth_h = dth(1:end-1);
    %%% probability
    % convolution method?
%     F = conv(dth, K_h, 'same') + conv(stim, K_s, 'same');
%     P = NL(F, beta);
    % design matrix method?
    % A = (convmtx((dth_h),length(K_h)));
    % B = (convmtx((stim_),length(K_s)));
    % F = A'*K_h' + B'*K_s' + base;
    
    F = conv_kernel(stim_,K_s) + conv_kernel(dth_h,K_h) + base;

    F = F(1:length(dth_));
    F(1:length(K_h)) = 0;
    P = NL(F,beta);

%%% log-likelihood
    y_1 = find(dth_ == 1);
    y_0 = find(dth_ == 0);
    ll = sum(log(P(y_1))) + sum(log(1-P(y_0)));
%     ll = sum(dth.*(log(P(y_logical))) + (1-dth).*log(1-P(y_logical)));
    NLL = -ll;
%     ll = sum(-dth.*P + log(P));%−nλ+tlnλ.
end

function [F] = conv_kernel(X,K)
%%%
% using time series X and kernel K and compute convolved X with K
% this takes care of padding and the time delay
% note that K is a causal vector with zero-time in the first index
%%%
    %%% change padding
    % pad_l = floor(length(K)/2);
    % padding = zeros(1,pad_l) + mean(X);  %padding vector
    % Fp = conv([padding, X, padding], K, 'same');  %convolvultion of vectors
    % F = Fp(pad_l+1:length(X)+pad_l);  %return the right size
    
    %%% old code
    pad_l = floor(length(K)/2);
    padding = zeros(1,pad_l);% + mean(X);  %padding vector
    Fp = conv([padding, X, padding], K, 'same');  %convolvultion of vectors
    F = Fp(1:length(X));  %return the right size
end