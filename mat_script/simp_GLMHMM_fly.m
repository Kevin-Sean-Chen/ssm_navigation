% simple_GLMHMM_fly
%%% run simple GLM-HMM test for fly navigation data
%%% input is biniarized odor encounter
%%% output is Bernoulli behavioral choices
%%% aim to plot stats in space and time!
%
%%% NEXT STEPS:
%%% % include driven state transitions
%%% % explore rawer emissions, such as angular change and speed
%%% % test across environments/data

%% load data
load('C:\Users\kevin\Yale University Dropbox\users\mahmut_demir\data\Smoke Navigation Paper Data\ComplexPlumeNavigationPaperData.mat')
full_data = ComplexPlume.Smoke.expmat;

%% build Data structure
%%% unpacking
trjNum = full_data(:,1);
signal = full_data(:,13);
stops = full_data(:,39);
turns = full_data(:,40);
x_smooth = full_data(:,32);
y_smooth = full_data(:,33);
speed_smooth = full_data(:,31);

signal(isnan(signal)) = 0;  % remove nans

%%% process actions and signal
diff_stop = diff(stops);
stops_time = stops*0;
% stops_time(find(diff_stop<0)) = 1;  % walking
stops_time(find(diff_stop>0)) = 1;  % stopping (BTA has rising shape)

bin_signal = signal*0;
bin_signal(find(signal>3)) = 1;
diff_signal = diff(bin_signal);
signal_time = diff_signal*0;
signal_time(find(diff_signal>0)) = 1;

%% make some prior
down_samp = 2;  % every 3 x 0.011 seconds
kernel_window = 1.5;  % in seconds
nb = 4;
cosBasis = makeRaisedCosBasis(nb, [0, kernel_window/(down_samp*0.011)/2.], 30);
lk = size(cosBasis,1);
figure; plot(cosBasis)

%% stack as tracks
ntracks = length(unique(trjNum));
list_tracks = unique(trjNum);
clear Data
Data(length(ntracks)) = struct();
di = 1;
for nn = 1:400 %ntracks
    pos = find(trjNum==list_tracks(nn));
    pos = pos(1:down_samp:end);
    Data(di).act = stops_time(pos); %stops
    Data(di).stim = bin_signal(pos); %signal(pos);  %signal_time(pos); %
    Data(di).lambda = .0;%.1;
    Data(di).Basis = cosBasis;
    mask = true(length(pos),1);
    mask(1:floor(lk/2)) = false; mask(end-floor(lk/2):end) = false;
    Data(di).mask = mask;
    Data(di).x_smooth = x_smooth(pos);
    Data(di).y_smooth = y_smooth(pos);
    Data(di).speed_smooth = speed_smooth(pos);
    di = di + 1;
end

%% then extract complete data vectors for EM
xx = [extractfield(Data,'stim')];  % input
yy = [extractfield(Data,'act')];  % observation
mask = [extractfield(Data,'mask')];  % mask for LL
mask = cat(1,mask{:})';

data_x = [extractfield(Data,'x_smooth')]; 
data_y = [extractfield(Data,'y_smooth')]; 
data_speed = [extractfield(Data,'speed_smooth')]; 

%% Observation and input
% Set parameters: transition matrix and emission matrix
nStates = 2; % number of latent states
nX = 7;  % number of input dimensions (i.e., dimensions of regressor)
nY = 1;  % number of output dimensions 
nT = length(yy); % number of time bins
loglifun = @logli_fly;  % log-likelihood function

% Set transition matrix by sampling from Dirichlet distr
alpha_diag = 10; %25 % concentration param added to diagonal (higher makes more diagonal-dominant)
alpha_full = 5;  % concentration param for other entries (higher makes more uniform)
G = gamrnd(alpha_full*ones(nStates) + alpha_diag*eye(nStates),1); % sample gamma random variables
A0 = G./repmat(sum(G,2),1,nStates); % normalize so rows sum to 1
% A0 = [0.99,0.01; 0.01,0.99];

% sticky priors
alpha = 1.;  % Dirichlet shape parameter as a prior
kappa = .5;  % upweighting self-transition for stickiness

% Set linear weights & output noise variances
wts0 = rand(nY,nX,nStates); % parameters for the mixture-VonMesis behavioral model
%%% Parameters: M, m, base, alpha
wts0(1,:,1) = [.9, .1, 0, randn(1,nb)*1]; %single mGLM
wts0(1,:,2) = [.6, .4, 0, randn(1,nb)*1];
% wts0(1,:,3) = [.9, .1, 0, randn(1,nB)*10];

% Build struct for initial params
mmhat = struct('A',A0, 'wts',wts0, 'loglifun',loglifun, 'basis',cosBasis, 'lambda',[.0]);

%% call inference procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug
[logli] = logli_fly(mmhat, xx, yy, mask);

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
    % [logp,gams,xis,xisum,logcs] = runFB_GLMHMM_xi(mmhat,xx,yy,mask);
    logpTrace(jj) = logp;
   
    % --- run M step  -------
    
    % Update transition matrix
    mmhat.A = (alpha-1 + xisum) ./ (nStates*(alpha-1) + sum(xisum,2)); % normalize each row to sum to 1
    
    % Update model params
    mmhat = runMstep_fly(mmhat, xx, yy, gams, mask);
    
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

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot some results
cols = ['k','r'];
stateK = 1;
flip = 1;
x = squeeze(mmhat.wts(:,:,stateK));
M = x(1);
m = x(2);
base = x(3);
alpha_s = flip*x(4:7);

K_s = alpha_s*cosBasis';
xv = -10:.1:10;

figure()
subplot(121); plot([1:length(K_s)]*0.011*down_samp, K_s, cols(stateK), 'LineWidth',4); 
xlabel('time (s)'); ylabel('weights')
set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);
subplot(122); plot(xv, (M-m)./(1+exp(flip*-(xv+base))) + m, cols(stateK), 'LineWidth',4); 
xlabel('proejcted input'); ylabel('P(walk)')
set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);

%%
[logp,gams,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask);
pos_state1 = find(gams(1,:)>0.5);
pos_state2 = find(gams(2,:)>0.5);
figure;
plot(data_x(pos_state1), data_y(pos_state1), 'k.'); hold on
plot(data_x(pos_state2), data_y(pos_state2), 'r.');

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% inference ll and M-step
function [P] = NL(F,M,m)
    P = (M-m)./(1+exp(-F)) + m;
    % P = 1./(1+exp(-F));
end

function [logli] = logli_fly(mm, xx, yy, mask)

% Compute log-likelihood term under a mixture of von Mesis model
%
% Inputs
% ------
%   mm [struct] - model structure with params 
%      .wts   [1 len(param) K] - per-state parameters for the model
%      .basis [len(alpha) len(kernel)] - basis functions used for the kernel
%      .lambda -scalar for regularization of the logistic fit
%    xx [1 T] - inputs (time series of sensory signal)
%    yy [1 T] - outputs (time series of behavioral markers)
%
% Output
% ------
%  logpy [T K] - loglikelihood for each observation for each state
    
    %%% loading parameters
    THETA = mm.wts;
    Basis = mm.basis;
    lambda = mm.lambda;
    act = yy;
    stim = xx;
    
    K = size(THETA,3);
    lt = length(yy);
    logli = zeros(lt, K);
    for k = 1:K
        %%% Assume we parameterize in such way first
        M = THETA(1,1,k);          % Max rate
        m = THETA(1,2,k);          % min rate
        base = THETA(1,3,k);       % baseline in sigmoid
        alpha_s = THETA(1,4:7,k);  % kernel for sensory input (weights on kerenl basis)

        %%% construct as kernel
        K_s = (alpha_s * Basis');  % sensory kernel

        %%% turning decision
        filt_stim = conv_kernel(stim, K_s);
        P = NL(filt_stim + 0 + base, M, m);
        % P = P*0.011;  % per time probability

        %%% Bernoulli log-likelihood
        epsilon = 1e-15;  % Small value to avoid log(0)
        P = max(min(P, 1 - epsilon), epsilon);
        logli(:,k) = [mask.* ((log(P).*act) + ((1-act).*log(1-P))) - lambda*sum((K_s - 0).^2)*0]';  % can later add regularization...
    end
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
gamnrm = gams./(sum(gams,2)+1e-10);  
nStates = size(gams,1);

%%% loading parameters
Basis = mm.basis;
lambda = mm.lambda;
act = yy;
stim = xx;
nb = size(Basis,2);

opts = optimset('display','iter'); % opts.Algorithm = 'sqp';
UB = [1,  1, 100, ones(1,nb*1)*100];
LB = [.0, .0, -100, -ones(1,nb*1)*100];
prs0 = [.9, .1, 0, randn(1,nb*1)];%

for jj = 1:nStates
    lfun = @(x)nll_fly(x, act, stim, gamnrm(jj,:), Basis, lambda, mask);
    prs0 = mm.wts(:,:,jj); %+ mm.wts(:,:,jj).*(2*(rand(1,length(UB))-0.5))*0.5;  %from last time!
    [x,fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);
    % [x,fval,exitflag,output,grad,hessian] = fminunc(lfun,prs0, opts);
    mm.wts(:,:,jj) = x; % weighted optimization
end

end

function [NLL] = nll_fly(THETA, act, stim, gams, cosBasis, lambda, mask)
    
    %%% unpacking
    base = THETA(1);
    M = abs(THETA(2));
    m = abs(THETA(3));
    %%% hakking proability
    if M<m
        [M, m] = deal(m, M);
    end
    nb = size(cosBasis, 2);
    alpha_s = THETA(3+1:3+nb);
    % alpha_a = THETA(3+nb+1:end);

    %%% construct filters
    K_s = (alpha_s*cosBasis');
    
    % design matrix method?
    F = conv_kernel(stim',K_s) + base*1;
    F = F(1:length(act));
    F(1:length(K_s)) = 0;
    P = NL(F, M, m);
    % P = P*0.011;  % per time probability

    %%% log-likelihood
    epsilon = 1e-15;  % Small value to avoid log(0)
    P = max(min(P, 1 - epsilon), epsilon)';
    
    %%% the Bernoulli way
    ll = sum(  mask.*gams.* (log(P).*act + (1-act).*log(1-P))  ) - lambda*(1*sum((K_s - 0).^2));
    % ll = sum(log(P(y_1))) + sum(log(1-P(y_0)));% - lambda*norm(K_a);

    NLL = -ll;
end
