% demo_analytic_GLMHMM
%%% serve as ground-truth to varify the inference procedure
%%% used to demo analytic and gradient method for inferring true parameters

%% generate hidden states
lt = 50000;  % length of simulation
nStates = 2;  % two hidden states for now
% Set transition matrix by sampling from Dirichlet distr
alpha_diag = 25;  % concentration param added to diagonal (higher makes more diagonal-dominant)
alpha_full = 5;  % concentration param for other entries (higher makes more uniform)
G = gamrnd(alpha_full*ones(nStates) + alpha_diag*eye(nStates),1); % sample gamma random variables
A0 = G./repmat(sum(G,2),1,nStates); % normalize so rows sum to 1
A0 = [0.99,0.01; 0.02,0.98];
% A0 = eye(2);
A0 % Markov transition
mc = dtmc(A0);
X = simulate(mc, lt);
alpha = 1;

%% define variables
nB = 4;  % number of basis function for the kernel
[cosBasis, tgrid, basisPeaks] = makeRaisedCosBasis(nB, [0, 10], 1.3); % basis function
%%% true params
alpha_h1 = [-4:-1]*.1;  % angle history kernel coefficient
alpha_dc1 = [1,4,-2,-1]*.5;  % dC kernel coefficient
base1 = -.5;  % baseline rate
sig1 = .1;

alpha_h2 = [-4,-2,-1,-1]*.3;  % angle history kernel coefficient
alpha_dc2 = [4,1,-2,-1]*.1;  % dC kernel coefficient
base2 = .5;  % baseline rate
sig2 = .2;

%%% construct as kernel
K_h1 = fliplr(alpha_h1*cosBasis');  % dth kernel
K_dc1 = fliplr(alpha_dc1*cosBasis');  % dC kernel
K_h2 = fliplr(alpha_h2*cosBasis');  % dth kernel
K_dc2 = fliplr(alpha_dc2*cosBasis');  % dC kernel

%% generate data
r2d = 1;%180/pi;
lc = 1;  %length of smoothing
dC = conv(randn(1,lt),ones(1,lc),'same')/lc*2;  % dC stimulus vector
dth = zeros(1,lt);
turns = zeros(1,lt);
F = dth*0;
pad = length(K_h1);
for tt=pad+1:lt
    if X(tt) == 1
        K_h = K_h1; K_dc = K_dc1; base = base1; sig = sig1;
    elseif X(tt) == 2
        K_h = K_h2; K_dc = K_dc2; base = base2; sig = sig2;
    end
    dth(tt) = dC(tt-pad+1:tt)*K_dc' + 1*(dth(tt-pad:tt-1))*K_h' + base + randn(1)*sig;  % linear filtering
end

allas = dth*r2d;
alldCs = dC;

figure; hist(allas,100)

%% load training data
wind = 1:lt;
yy = allas(wind);
xx = alldCs(wind); 

mask = true(1,length(yy));

%% call inference procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Observation and input
% Set parameters: transition matrix and emission matrix
nStates = 2; % number of latent states
nX = nB*2+2;  % number of input dimensions (i.e., dimensions of regressor)
nY = 1;  % number of output dimensions 
nT = length(yy); % number of time bins
loglifun = @logli_mgGLM;  % ground truth model log-likelihood function; mixture-Gaussian GLMs

% Set transition matrix by sampling from Dirichlet distr
% alpha_diag = 25;  % concentration param added to diagonal (higher makes more diagonal-dominant)
% alpha_full = 10;  % concentration param for other entries (higher makes more uniform)
% G = gamrnd(alpha_full*ones(nStates) + alpha_diag*eye(nStates),1); % sample gamma random variables
% A0 = G./repmat(sum(G,2),1,nStates); % normalize so rows sum to 1
A0 = A0;

% basis function
nB = 4;
[cosBasis, tgrid, basisPeaks] = makeRaisedCosBasis(nB, [0, 10], 1.3);

% Set linear weights & output noise variances
% wts0 = [10, randn(1,nB)*10, 10, 25, 10, 25, 5, 1.]; 
wts0 = rand(nY,nX,nStates); % parameters for the mixture-VonMesis behavioral model
wts0(1,:,1) = [alpha_dc1, alpha_h1, base1, sig1]; %single mGLM
wts0(1,:,2) = [alpha_dc2, alpha_h2, base2, sig2];
% wts0 = wts0 + rand(nY, nX, nStates);

% Build struct for initial params
mmhat = struct('A',A0,'wts',wts0,'loglifun',loglifun,'basis',cosBasis,'lambda',.0, 'Mstepfun',@runMstep_GTmVM);

%% debug
% OLS_glm(xx, yy, ones(1,lt), mask, cosBasis);
% [logli] = logli_mgGLM(mmhat, xx, yy, mask);

%% Set up variables for EM
maxiter = 50;
EMdisplay = 2;
logpTrace = zeros(maxiter,1); % trace of log-likelihood
dlogp = inf; % change in log-likelihood
logpPrev = -inf; % prev value of log-likelihood
jj = 1; % counter

while (jj <= maxiter) && (dlogp>1e-3)
    
    % --- run E step  -------
    [logp,gams,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask); % run forward-backward
    logpTrace(jj) = logp;

    % --- run M step  -------
    
    % Update transition matrix
    mmhat.A = (alpha-1 + xisum) ./ (nStates*(alpha-1) + sum(xisum,2)); % normalize each row to sum to 1
    
    % Update model params
    % mmhat = runMstep_mgGLM(mmhat, xx, yy, gams, mask);
    mmhat = runMstep_fly(mmhat, xx, yy, gams, mask);
    %mmhat.Mstepfun(mmhat,xx(:,mask),yy(:,mask),gams(:,mask));
    
    % ---  Display progress ----
    if mod(jj,EMdisplay)==0
        fprintf('EM iter %d:  logli = %-.6g\n',jj,logp);
    end
    
    % Update counter and log-likelihood change
    jj = jj+1;  
    dlogp = logp-logpPrev; % change in log-likelihood
    logpPrev = logp; % previous log-likelihood

    if dlogp<-1e-6
        warning('Log-likelihood decreased during EM!');
        fprintf('dlogp = %.5g\n', dlogp);
    end

end
jj = jj-1;


%% compare to gound-truth
stateK = 2;
x = squeeze(mmhat.wts(:,:,stateK));
alpha_s = x(1:4);       % kernel for dth angle history kernel (weights on kerenl basis)
alpha_h = x(5:8);      % kernel for dC transitional concentration difference (weights on kerenl basis)
base_ = x(end);          % baseline rate

K_h_rec = alpha_h*cosBasis';
K_dc_rec = alpha_s*cosBasis';

figure()
subplot(121)
plot(fliplr(K_h_rec)); hold on; plot(K_h1,'--'); plot(K_h2,'--');
subplot(122)
plot(fliplr(K_dc_rec)); hold on; plot(K_dc1,'--'); plot(K_dc2,'--');

[aa,bb] = max( gams ,[], 1 );
figure;
plot(allas(:)); hold on
% plot(smooth((bb(wind)-1)*100,10))
plot((bb(:)-1)*3)

%% inference ll and M-step
function [logli] = logli_mgGLM(mm, xx, yy, mask)

% Compute log-likelihood term under a mixture of von Mesis model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%      .wts   [1 len(param) K] - per-state parameters for the model
%      .basis [len(alpha) len(kernel)] - basis functions used for the kernel
%      .lambda -scalar for regularization of the logistic fit
%    xx [2 T] - inputs (time series of dc,dcp, and dth)
%    yy [1 T] - outputs (dth angles)
%
% Output
% ------
%  logpy [T K] - loglikelihood for each observation for each state
    
    %%% loading parameters
    THETA = mm.wts;
    Basis = (mm.basis); % lk x nb
    lk = size(Basis,1);
    
    K = size(THETA,3);
    lt = length(yy);
    logli = zeros(lt, K);
    for k = 1:K
        %%% Assume we parameterize in such way first
        alpha_dc = THETA(1,1:4,k);      % kernel for dth angle history kernel (weights on kerenl basis)
        alpha_h = THETA(1,5:8,k);       % kernel for dC transitional concentration difference (weights on kerenl basis)
        base = THETA(1,9,k);            % baseline
        sig = (THETA(1,10,k))^2+0.00001;       % sigma

        %%% construct design matrix
        % X = hankel(xx(1:lk), xx(lk:end))';  % (T-lk+1) x lk
        % y_hist = [0 yy(1:end-1)];
        % H = hankel(y_hist(1:lk), y_hist(lk:end))';  % (T-lk+1) x lk
        % X = [X; rand(lk-1, lk)*0];
        % H = [H; rand(lk-1, lk)*0];
        % Xf = X * Basis;  % T x nb
        % Hf = H * Basis;
        % Xf = [Xf  Hf  ones(size(X,1),1)];  % add offset
        % y_hat = Xf * [alpha_dc, alpha_h,  base]';
        %%% sig = mean((y_hat'-yy).^2);

        %%%%
        %%% FIGURE OUT: the difference between matrix and conv_kernel method!
        %%%%
        %%% construct filters
        K_s = (alpha_dc*Basis');
        K_h = (alpha_h*Basis');

        % design matrix method?
        F = conv_kernel(xx(2:end),K_s) + conv_kernel(yy(1:end-1),K_h) + base*1;
        logp = -1/(2*sig)*(F - yy(2:end)).^2 - log(sig);

        %%% marginal probability
        logli(2:end,k) = logp;%
        % logli(:,k) = -0.5*mask.*(y_hat' - yy).^2/sig - log(sig); 
    end
end

function mm = runMstep_mgGLM(mm, xx, yy, gams, mask)
% mm = runMstep_LinGauss(mm,xx,yy,gams)
%
% Run m-step updates for Gaussian observation model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%        .wts  [1 K] - per-state slopes
%        .vars [1 K] - per-state variances
%    xx [d T] - input (design matrix)
%    yy [1 T] - outputs
%  gams [K T] - log marginal sate probabilities given data
%
% Output
% ------
%  mmnew - new model struct

% normalize the gammas to sum to 1
gamnrm = gams./(sum(gams,2)+1e-10);  
nStates = size(gams,1);

%%% loading parameters
Basis = (mm.basis);

%%% state wise OLS
for jj = 1:nStates
    x = OLS_glm(xx, yy, gamnrm(jj,:), mask, Basis);
    mm.wts(:,:,jj) = x;
end

end

function beta = OLS_glm(xx, yy, gams, mask, Basis)
    lk = size(Basis,1);
    wind = 1:length(xx)-lk+1;
    X = hankel(xx(1:lk), xx(lk:end))';  % T x lk
    y_hist = [0 yy(1:end-1)];
    H = hankel(y_hist(1:lk), y_hist(lk:end))';  % (T-lk+1) x lk
    Xf = X * Basis;  % T x nb
    Hf = H * Basis;
    Xf = [Xf  Hf  ones(size(Xf,1),1)];  % add offset
    % W = diag(mask(1:end-lk+1).*gams(1:end-lk+1));  %%% reweighting
    v = mask(wind).*gams(wind);
    W = spdiags(v', 0, length(v), length(v));
    XXT = Xf'* W * Xf;
    % XXT = Xf'*Xf;
    Xy = Xf' * (v.*yy(wind))';
    beta = XXT \ Xy;  %%% OLS
    % beta = ones(1,9);
    sig = mean(((Xf*beta)' - yy(wind)).^2)^0.5;
    beta = [beta; sig];
end

function mm = runMstep_fly(mm, xx, yy, gams, mask)
%
% Run m-step updates for GLM-like observation model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%        .wts  [1 K] - per-state slopes
%        .vars [1 K] - per-state variances
%    xx [1 T] - input (design matrix)
%    yy [1 T] - outputs
%  gams [K T] - log marginal sate probabilities given data
%
% Output
% ------
%  mmnew - new model struct

% normalize the gammas to sum to 1
gamnrm = gams./(sum(gams,2)+1e-20);  
nStates = size(gams,1);

%%% loading parameters
Basis = mm.basis;
lambda = mm.lambda;
nb = size(Basis, 2);
act = yy;
stim = xx;
opts = optimset('display','iter'); %
UB = [ones(1,nb*2)*100, 100, 100];
LB = [-ones(1,nb*2)*100, -100, .1];

for jj = 1:nStates
    lfun = @(x)nll_fly(x, act, stim, gamnrm(jj,:), Basis, lambda, mask);
    prs0 = mm.wts(:,:,jj); 
    [x,fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);
    % [x,fval,exitflag,output,grad,hessian] = fminunc(lfun,prs0, opts);
    mm.wts(:,:,jj) = x; % weighted optimization
end

end

function [NLL] = nll_fly(THETA, act, stim, gams, cosBasis, lambda, mask)
    
    %%% unpacking
    nb = size(cosBasis, 2);
    alpha_s = THETA(1:nb);
    alpha_h = THETA(nb+1:nb*2);
    base = THETA(end-1);
    sigma2 = THETA(end)^2;
    % alpha_a = THETA(3+nb+1:end);

    %%% construct filters
    K_s = (alpha_s*cosBasis');
    K_h = (alpha_h*cosBasis');
    % K_s = Laguerre(alpha_s(1), alpha_s(2:end), size(cosBasis,1));
    
    % design matrix method?
    F = conv_kernel(stim(2:end),K_s) + conv_kernel(act(1:end-1),K_h) + base*1;
    logp = -1/(2*sigma2)*(F - act(2:end)).^2 - log(sigma2);
    
    %%% the Bernoulli way
    ll = sum(  mask(2:end).*gams(2:end).* logp  );

    NLL = -ll;
end
