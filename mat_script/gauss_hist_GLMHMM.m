% gaussian_history_GLMHMM
%%% run GLM-HMM test for GLM output speed for fly navigation data
%%% input is biniarized odor encounter
%%% output is a GLM with Gaussian output and autoregressive signal
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
speed_smooth = full_data(:,31);%11 31
dtheta_smooth = full_data(:,12);%14 35

signal(isnan(signal)) = 0;  % remove nans
speed_smooth(speed_smooth>30) = 30;%mean(speed_smooth);
speed_smooth(speed_smooth==0) = .1;

%%% process actions and signal
diff_stop = diff(stops);
stops_time = stops*0;
% stops_time(find(diff_stop<0)) = 1;  % walking
stops_time(find(diff_stop>0)) = 1;  % stopping (BTA has rising shape)

bin_signal = signal*0;
bin_signal(find(signal>1)) = 1;
diff_signal = diff(bin_signal);
signal_time = diff_signal*0;
signal_time(find(diff_signal>0)) = 1;

%% make some prior
down_samp = 1;  % every 3 x 0.011 seconds
kernel_window = 1.5;  % in seconds
nb = 4;
cosBasis = makeRaisedCosBasis(nb, [0, kernel_window/(down_samp*0.011)/2.], 25*2);
lk = size(cosBasis,1);
figure; plot(cosBasis)

%% stack as tracks
ntracks = length(unique(trjNum));
list_tracks = unique(trjNum);
clear Data
Data(length(ntracks)) = struct();
di = 1;
for nn = 1:ntracks-1
    pos = find(trjNum==list_tracks(nn));
    pos = pos(1:down_samp:end);
    Data(di).act = speed_smooth(pos); %stops_time(pos); %stops
    Data(di).stim = signal_time(pos); %bin_signal(pos); %signal(pos);  %
    Data(di).lambda = .0;%.1;
    Data(di).Basis = cosBasis;
    mask = true(length(pos),1);
    mask(1:floor(lk/2)) = false; mask(end-floor(lk/2):end) = false;
    Data(di).mask = mask;
    Data(di).x_smooth = x_smooth(pos);
    Data(di).y_smooth = y_smooth(pos);
    Data(di).speed_smooth = speed_smooth(pos);
    Data(di).dtheta_smooth = dtheta_smooth(pos);
    di = di + 1;
end

%% train and test sets
Data_train = Data(1:500);
Data_test = Data(500:end);

%% then extract complete data vectors for EM
xx = [extractfield(Data_train,'stim')];  % input
yy = [extractfield(Data_train,'act')];  % observation
mask = [extractfield(Data_train,'mask')];  % mask for LL
mask = cat(1,mask{:})';

data_x = [extractfield(Data_train,'x_smooth')]; 
data_y = [extractfield(Data_train,'y_smooth')]; 
data_speed = [extractfield(Data_train,'speed_smooth')]; 
data_dtheta =  [extractfield(Data_train,'dtheta_smooth')];

%% quick OLS checkl
% lk = 200;
% x_ =xx(pos_state2); y_=yy(pos_state2);
% % X = [convmtx(xx(lk:end),lk)];    % (n x T)
% X = hankel(x_(1:lk), x_(lk:end));
% X = [X; ones(1,size(X,2))];
% XXT_inv = (X * X'); % (n x n)
% Xy = X * y_(1:end-lk+1)';           % (n x 1)
% beta = XXT_inv \ Xy;   % (n x 1)
% figure
% kernel = (beta(2:end-1));
% plot([1:length(kernel)]*0.011*down_samp, kernel,'k-o','LineWidth',5); set(gcf, 'Color', 'w');
% xlabel('time lag (s)'); ylabel('weight'); set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);

%% Observation and input
% Set parameters: transition matrix and emission matrix
nStates = 3; % number of latent states
nX = nb*2+2;  % number of input dimensions (i.e., dimensions of regressor)
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
%%% Parameters: alpha, base, sigma
wts0(1,:,1) = [randn(1,nb)*1 randn(1,nb)*1 0, 1]; %single mGLM
wts0(1,:,2) = [randn(1,nb)*1 randn(1,nb)*1 0, 1];
wts0(1,:,3) = [randn(1,nb)*1 randn(1,nb)*1 0, 1]; %[.9, .1, 0, randn(1,nB)*10];

% Build struct for initial params
mmhat = struct('A',A0, 'wts',wts0, 'loglifun',loglifun, 'basis',cosBasis, 'lambda',[.0]);

%% call inference procedure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% debug
% [logli] = logli_fly(mmhat, xx, yy, mask);
% [NLL] = nll_fly(x, yy, xx, gams(2,:), cosBasis, 0, mask)
% [logp,gams,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask);

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
    % gams = gams./(sum(gams,1)+1e-20);
   
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
cols = ['k','r','b'];
stateK = 3;
flip = 1;
x = squeeze(mmhat.wts(:,:,stateK));
alpha_s = flip*x(1:nb);
alpha_h = x(nb+1:nb*2);
K_s = alpha_s*cosBasis';
K_h = alpha_h*cosBasis';
base = x(end-1);
% K_s = Laguerre(alpha_s(1), alpha_s(2:4), size(cosBasis,1));
xv = -10:.1:10;

figure()
subplot(121); plot([1:length(K_s)]*0.011*down_samp, K_s, cols(stateK), 'LineWidth',4); title('K_{odor}')
xlabel('time (s)'); ylabel('weights');set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);set(gcf, 'Color', 'w');
subplot(122); plot([1:length(K_h)]*0.011*down_samp, K_h, cols(stateK), 'LineWidth',4); title('K_{self}') 
xlabel('time (s)'); ylabel('weights');set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5);set(gcf, 'Color', 'w');


%% state predictions
[logp,gams_,xisum] = runFB_GLMHMM(mmhat,xx,yy,mask);
[aa,bb] = max( gams_ ,[], 1 );
pos_state1 = find(bb==1);
pos_state2 = find(bb==2);
pos_state3 = find(bb==3);

% pos_state1 = find(gams(1,:)>0.5);
% pos_state2 = find(gams(2,:)>0.5);
figure;
% plot(data_x(pos_state1), data_y(pos_state1), 'k.', 'MarkerFaceAlpha',.2); hold on
% plot(data_x(pos_state2), data_y(pos_state2), 'r.', 'MarkerFaceAlpha',.2);
scatter(data_x(pos_state1), data_y(pos_state1), 2, 'k', 'filled', 'MarkerFaceAlpha', 1, 'MarkerEdgeAlpha', 1); hold on
scatter(data_x(pos_state2), data_y(pos_state2), 4, 'r', 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5);set(gcf, 'Color', 'w');
scatter(data_x(pos_state3), data_y(pos_state3), 2, 'b', 'filled', 'MarkerFaceAlpha', 0.5, 'MarkerEdgeAlpha', 0.5);

%%
figure;
subplot(211); h = histogram(data_speed(pos_state1),100); h.FaceColor = 'black'; xlim([0,29]);set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5)
subplot(212); h = histogram(data_speed(pos_state2),100); h.FaceColor = 'red'; set(gcf, 'Color', 'w');xlim([0,29]); set(gca, 'FontSize', 14, 'XColor', 'k', 'YColor', 'k', 'LineWidth', 1.5)
% set(gca, 'YScale', 'log');

%% simple cross-validation
reps = 20;
wind_test = 100000;
test_logp = zeros(1,reps);
xx_test = [extractfield(Data(end-500:end),'stim')];  % input
yy_test = [extractfield(Data(end-500:end),'act')];  % observation
mask_test = [extractfield(Data(end-500:end),'mask')];  % mask for LL
mask_test = cat(1,mask_test{:})';

for rr = 1:reps
    start_t = randi([1, length(xx_test)-wind_test]);
    x_samp = xx_test(start_t:start_t+wind_test);
    y_samp = yy_test(start_t:start_t+wind_test);
    m_samp = xx_test(start_t:start_t+wind_test);
    [logp,gams,xisum] = runFB_GLMHMM(mmhat,x_samp,y_samp,m_samp); % run forward-backward
    test_logp(rr) = logp;
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% inference ll and M-step
function [P] = NL(F,M,m)
    % P = (M-m)./(1+exp(-F)) + m;
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
    nb = size(Basis,2);
    
    K = size(THETA,3);
    lt = length(yy);
    logli = zeros(lt, K);
    for k = 1:K
        %%% Assume we parameterize in such way first
        alpha_s = THETA(1,1:nb,k);      % kernel for sensory input (weights on kerenl basis)
        alpha_h = THETA(1,nb+1:nb*2,k); % history kernel
        sigma2 = THETA(1,end,k)^2;      % sigmal variance for Gaussian
        base = THETA(1,end-1,k);       % baseline in sigmoid

        %%% construct as kernel
        K_s = (alpha_s * Basis');  % sensory kernel
        K_h = alpha_h * Basis';   % history kernel
        % K_s = Laguerre(alpha_s(1), alpha_s(2:end), size(Basis,1));

        %%% Gaussian regression
        filt_stim = conv_kernel(stim(2:end), K_s) + conv_kernel(act(1:end-1), K_h) + base;
        logp = -1/(2*sigma2)*(act(1:end-1) - filt_stim).^2 - log(sigma2);
        if K>1
            logli(2:end,k) = [mask(2:end).* logp - lambda*sum((K_s - 0).^2)*0]';  % can later add regularization...
        elseif K==1
            logli(2:end) = [mask(2:end).* logp - lambda*sum((K_s - 0).^2)*0]'; 
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
UB = [ones(1,nb*1)*100, 100, 100];
LB = [-ones(1,nb*1)*100, -100, .1];
prs0 = [.9, .1, 0, randn(1,nb*1)];%

for jj = 1:nStates
    lfun = @(x)nll_fly(x, act, stim, gamnrm(jj,:), Basis, lambda, mask);
    prs0 = mm.wts(:,:,jj); %+ mm.wts(:,:,jj).*(2*(rand(1,length(UB))-0.5))*0.5;  %from last time!
    % [x,fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(lfun,prs0,[],[],[],[],LB,UB,[],opts);
    [x,fval,exitflag,output,grad,hessian] = fminunc(lfun,prs0, opts);
    mm.wts(:,:,jj) = x; % weighted optimization
end

end

function [NLL] = nll_fly(THETA, act, stim, gams, cosBasis, lambda, mask)
    
    %%% unpacking
    nb = size(cosBasis, 2);
    base = THETA(end-1);
    alpha_s = THETA(1:nb);
    alpha_h = THETA(nb+1:nb*2);
    sigma2 = THETA(end)^2;
    % alpha_a = THETA(3+nb+1:end);

    %%% construct filters
    K_s = (alpha_s*cosBasis');
    K_h = alpha_h*cosBasis';
    % K_s = Laguerre(alpha_s(1), alpha_s(2:end), size(cosBasis,1));
    
    % design matrix method?
    F = conv_kernel(stim(2:end), K_s) + conv_kernel(act(1:end-1), K_h) + base*1;
    F = F(1:length(act(2:end)));
    logp = -1/(2*1)*(F - act(2:end)).^2 - log(1);
    
    %%% the Bernoulli way
    ll = sum(  mask(2:end).*gams(2:end).* logp  ) - lambda*(1*sum((K_s - 0).^2));

    NLL = -ll;
end

function [Kt] = Laguerre(lambda, beta, kernel_length)
    % Laguerre generates a Laguerre basis function
    %
    % Parameters:
    % lambda       : The decay parameter for the exponential term
    % beta         : Coefficients vector for the basis function
    % kernel_length: Length of the output kernel
    %
    % Returns:
    % Kt           : Generated Laguerre basis function

    k = length(beta);                % Number of coefficients
    tt = 0:kernel_length-1;          % Time vector from 0 to kernel_length-1
    Kt = zeros(size(tt));            % Initialize the output kernel

    for kk = 1:k
        Kt = Kt + lambda * exp(-lambda * tt) .* beta(kk) .* (lambda * tt).^(kk - 1);
    end
end