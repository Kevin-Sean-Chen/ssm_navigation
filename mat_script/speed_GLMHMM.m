% speed_GLMHMM_fly
%%% run speed GLM-HMM test for fly navigation data
%%% input is biniarized odor encounter
%%% output is mixture of gamma for navigation speed
%%% aim to plot stats in space and time!
%
%%% NEXT STEPS:
%%% % include driven state transitions
%%% % explore rawer emissions, such as angular change and speed
%%% % test across environments/data

% clear
% clc
% rng(0)

%% load data
% load('C:\Users\kevin\Yale University Dropbox\users\mahmut_demir\data\Smoke Navigation Paper Data\ComplexPlumeNavigationPaperData.mat')
load('C:\Users\ksc75\Yale University Dropbox\users\mahmut_demir\data\Smoke Navigation Paper Data\ComplexPlumeNavigationPaperData.mat')
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

speed_smooth(speed_smooth>30) = 0;%mean(speed_smooth);
signal(isnan(signal)) = 0;  % remove nans

%%% process actions and signal
diff_stop = diff(stops);
stops_time = stops*0;
stops_time(find(diff_stop<0)) = 1;  % walking
% stops_time(find(diff_stop>0)) = 1;  % stopping (BTA has rising shape)

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
    Data(di).act = speed_smooth(pos); stops_time(pos); %stops
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
nX = 9; % number of input dimensions (i.e., dimensions of regressor)
nY = 1;  % number of output dimensions 
nT = length(yy); % number of time bins
loglifun = @logli_fly;  % log-likelihood function

% Set transition matrix by sampling from Dirichlet distr
alpha_diag = 25; %25 % concentration param added to diagonal (higher makes more diagonal-dominant)
alpha_full = 5;  % concentration param for other entries (higher makes more uniform)
G = gamrnd(alpha_full*ones(nStates) + alpha_diag*eye(nStates),1); % sample gamma random variables
A0 = G./repmat(sum(G,2),1,nStates); % normalize so rows sum to 1
% A0 = [0.99,0.01; 0.01,0.99];

% sticky priors
alpha = 1.;  % Dirichlet shape parameter as a prior
kappa = .5;  % upweighting self-transition for stickiness

% Set linear weights & output noise variances
wts0 = rand(nY,nX,nStates); % parameters for the mixture-VonMesis behavioral model
%%% Parameters: a1,b1,a2,b2,alpha(1,nb),base
wts0(1,:,1) = [rand(1,4), randn(1,nb)*1, 0]; %single mGLM
wts0(1,:,2) = [rand(1,4), randn(1,nb)*1, 0];
% wts0(1,:,3) = [.9, .1, 0, randn(1,nB)*10];

% Build struct for initial params
mmhat = struct('A',A0, 'wts',wts0, 'loglifun',loglifun, 'basis',cosBasis, 'lambda',[.0]);

%% call inference procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug
% [logli] = logli_fly(mmhat, xx, yy, mask);
% [NLL] = nll_fly(ones(1,nX), yy, xx, gams(2,:), cosBasis, 0, mask);

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
%% plot results for driven mixture
cols = ['k','r'];
stateK = 1;
flip = 1;
x = squeeze(mmhat.wts(:,:,stateK));
a1 = x(1);
b1 = x(2);
a2 = x(3);
b2 = x(4);
M = 1; m = 0;  %%%% without fitting max-min probablity
alpha_s = flip*x(5:8);
base = x(9);

K_s = alpha_s*cosBasis';
xv = -10:.1:10;

figure()
subplot(121); plot([1:length(K_s)]*0.011*down_samp, K_s, cols(stateK), 'LineWidth',4); 
xlabel('time (s)'); ylabel('weights')
set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);
subplot(122); plot(xv, (M-m)./(1+exp(flip*-(xv+base))) + m, cols(stateK), 'LineWidth',4); 
xlabel('proejcted input'); ylabel('P(walk)')
set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);

%% state predictions in space
[logp,gams_,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask);
[aa,bb] = max( gams_ ,[], 1 );
pos_state1 = find(bb==1);
pos_state2 = find(bb==2);
% pos_state1 = find(gams(1,:)>0.5);
% pos_state2 = find(gams(2,:)>0.5);
figure;
% plot(data_x(pos_state1), data_y(pos_state1), 'k.', 'MarkerFaceAlpha',.2); hold on
% plot(data_x(pos_state2), data_y(pos_state2), 'r.', 'MarkerFaceAlpha',.2);

scatter(data_x(pos_state1), data_y(pos_state1), 10, 'k', 'filled', 'MarkerFaceAlpha', 1, 'MarkerEdgeAlpha', 1); hold on
scatter(data_x(pos_state2), data_y(pos_state2), 15, 'r', 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5); hold on

%% matching observations
figure;
subplot(211);
stateK = 2; x = squeeze(mmhat.wts(:,:,stateK));
a1 = x(1); b1 = x(2); a2 = x(3); b2 = x(4);
[cc, bb] = histcounts(data_speed(pos_state1), 100); bb = bb(2:end);
gam_1 = gampdf(bb, a1, b1); gam_2 = gampdf(bb, a2, b2);
bar(bb,cc/sum(cc)); hold on
plot(bb, gam_1/sum(gam_1))
subplot(212);
stateK = 2; x = squeeze(mmhat.wts(:,:,stateK));
a1 = x(1); b1 = x(2); a2 = x(3); b2 = x(4);
[cc, bb] = histcounts(data_speed(pos_state2), 100); bb = bb(2:end);
gam_1 = gampdf(bb, a1, b1); gam_2 = gampdf(bb, a2, b2);
bar(bb,cc/sum(cc)); hold on
plot(bb, gam_2/sum(gam_2))

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% inference ll and M-step
function [P] = NL(F)
    P = 1./(1+exp(-F));
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
        %%% a1,b1,a2,b2,gamma,alpha(1,nb),base
        %%% Assume we parameterize in such way first
        a1 = THETA(1,1,k);         % gamma shape and scale
        b1 = THETA(1,2,k);
        a2 = THETA(1,3,k);
        b2 = THETA(1,4,k);
        alpha_s = THETA(1,5:8,k);  % kernel for sensory input (weights on kerenl basis)
        base = THETA(1,9,k);      % baseline choice

        %%% construct as kernel
        K_s = (alpha_s * Basis');  % sensory kernel

        %%% turning decision
        filt_stim = conv_kernel(stim, K_s);
        P = NL(filt_stim + 0 + base);
        % P = P*0.011;  % per time probability
        % P = max(0, min(1, P));  % for unconstrained optimization

        %%% driven-mixture Gamma log-likelihood
        epsilon = 1e-20;  % Small value to avoid log(0)
        P = max(min(P, 1 - epsilon), epsilon);

        pos = (~isinf(log(act)));
        ll_gamm1 = (a1-1)*log(act.*pos) - act/b1 - a1*log(b1) - gammaln(a1);
        ll_gamm2 = (a2-1)*log(act.*pos) - act/b2 - a2*log(b2) - gammaln(a2);
        if K>1
            logli(:,k) = mask.* (ll_gamm1.*P + ll_gamm2.*(1-P));
        elseif K==1
            logli(:) = mask.* (ll_gamm1.*P + ll_gamm2.*(1-P));;
        end

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
gamnrm = gams./(sum(gams,2)+1e-20);  
nStates = size(gams,1);

%%% loading parameters
Basis = mm.basis;
lambda = mm.lambda;
act = yy;
stim = xx;
nb = size(Basis,2);

opts = optimset('display','iter'); % opts.Algorithm = 'sqp';
%%% Parameters: a1,b1,a2,b2,alpha(1,nb),base
UB = [ones(1,4)*100, ones(1,nb*1)*100,  100];
LB = [ones(1,4)*.1, -ones(1,nb*1)*100, -100];
prs0 = [ones(1,4), randn(1,nb*1), 0];%

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
    a1 = THETA(1);         % gamma shape and scale
    b1 = THETA(2);
    a2 = THETA(3);
    b2 = THETA(4);
    alpha_s = THETA(5:8);  % kernel for sensory input (weights on kerenl basis)
    base = THETA(9);      % baseline choice

    %%% construct filters
    K_s = (alpha_s*cosBasis');
    
    % design matrix method?
    F = conv_kernel(stim',K_s) + base*1;
    F = F(1:length(act));
    P = NL(F);
    % P = P*0.011;  % per time probability
    % P = max(0, min(1, P));  % for unconstrained optimization

    %%% log-likelihood
    epsilon = 1e-20;  % Small value to avoid log(0)
    P = max(min(P, 1 - epsilon), epsilon)';
    
    %%% driven-mixture Gamma model
    pos = (~isinf(log(act)));
    ll_gamm1 = (a1-1)*log(act.*pos) - act/b1 - a1*log(b1) - gammaln(a1);
    ll_gamm2 = (a2-1)*log(act.*pos) - act/b2 - a2*log(b2) - gammaln(a2);
    ll = nansum(gams.* mask.* (ll_gamm1.*P + ll_gamm2.*(1-P)));
    NLL = -ll;
end
