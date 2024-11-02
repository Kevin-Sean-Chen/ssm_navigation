%% load data
load('ComplexPlumeNavigationPaperData.mat', 'ComplexPlume');
full_data = ComplexPlume.Smoke.expmat;
% full_data = ComplexPlumeWithBlocks.Smoke.expmat;

%% unpacking
trjNum = full_data(:,1);
signal = full_data(:,13);
stops = full_data(:,39);
turns = full_data(:,40);

% trjNum = full_data(:,1);
% signal = full_data(:,12);
% stops = full_data(:,37);
% turns = full_data(:,38);

%% process actions and signal
diff_stop = diff(stops);
stops_time = stops*0;
stops_time(find(diff_stop>0)) = 1;

% diff_stop = diff(turns);
% stops_time = stops*0;
% stops_time(find(diff_stop>0)) = 1;

bin_signal = signal*0;
bin_signal(find(signal>3)) = 1;
diff_signal = diff(bin_signal);
signal_time = diff_signal*0;
signal_time(find(diff_signal>0)) = 1;

%% make some prior
down_samp = 2;  % every 3 x 0.011 seconds
kernel_window = 2;  % in seconds
nb = 4;
cosbasis = makeRaisedCosBasis(nb, [0, kernel_window/(down_samp*0.011)/3], 15);
lk = size(cosbasis,1);
% figure; plot(cosbasis)

% test = Laguerre(.1, randn(4), 100);
% plot(test)

%% stack as tracks
ntracks = length(unique(trjNum));
list_tracks = unique(trjNum);
clear Data
Data(length(ntracks)) = struct();

for nn = 1:200 %ntracks
    pos = find(trjNum==list_tracks(nn));
    pos = pos(1:down_samp:end);
    Data(nn).act = stops_time(pos); %stops
    Data(nn).stim = bin_signal(pos); %signal
    Data(nn).lambda = 0.1;
    Data(nn).Basis = cosbasis;
    mask = true(1, length(pos));
    % mask(1:lk) = false; mask(end-lk:end) = false;
    Data(nn).mask = mask;
end

%% try GLM
lfun = @(x)group_nll(x, Data);  % objective function
opts = optimset('display','iter');
num_par = 3 + nb*2;
%%% base, M, m, alpha_s, alpha_a
UB = [100,  1,  1, ones(1,nb*2)*100];
LB = [-100, .0, .0, -ones(1,nb*2)*100];
prs0 = [0, .9, .1, randn(1,nb*2)];%
% c(x) <= 0 implies x(1) - x(2) >= 0
constraint = @(x) deal(x(3) - x(2), []);  % The inequality constraint
[x,fval,EXITFLAG,OUTPUT,LAMBDA,GRAD,HESSIAN] = fmincon(lfun,prs0,[],[],[],[],LB,UB,constraint,opts);  % constrained optimization

%% unpack the fitted parameters
base = x(1); M = x(2); m = x(3); alpha_s = x(3+1:3+nb); alpha_a = x(3+nb+1:end);

K_s = (alpha_s*cosbasis');
K_a = (alpha_a*cosbasis');
time_vec = [1:size(cosbasis,1)]*0.011*down_samp;
figure()
plot(time_vec,K_s); hold on
% plot(time_vec, K_a)
set(0, 'DefaultFigureColor', 'w'); set(gca,'FontSize',20)
xlabel('time lag (s)'); ylabel('weights'); title('walk kernel')

%% run GLM-HMM!
% nll(prs0, Data(nn).act, Data(nn).stim, Data(nn).Basis, Data(nn).lambda, Data(nn).mask)
%  % design matrix method?
%     % A = (convmtx((act),length(K_a)));
%     % B = (convmtx((stim),length(K_s)));
%     % size(A)
%     % size(K_a)
%     % F = A*K_a' + B*K_s' + base;
%     % F = F(1:length(act));
%     % F(1:length(K_a)) = 0;
%     % % F = F.*mask;
%     % P = NL(F, M, m);

conv_kernel([1:10],[1,1,1,1])
%% functions here
function [gnll] = group_nll(THETA, data)
    gnll = 0;
    ntracks = length(data);
    for nn = 1:ntracks
        Z = length(data(nn).act);  %normalization for nLL per time!
        gnll = gnll + nll(THETA, data(nn).act, data(nn).stim, data(nn).Basis, data(nn).lambda, data(nn).mask)/Z; % sum across tracks normalize by time
    end
end

function [NLL] = nll(THETA, act, stim, cosBasis, lambda, mask)
    
    %%% unpacking
    base = THETA(1);
    M = THETA(2);
    m = THETA(3);
    nb = size(cosBasis, 2);
    alpha_s = THETA(3+1:3+nb);
    alpha_a = THETA(3+nb+1:end);

    %%% construct filters
    K_s = (alpha_s*cosBasis');
    K_a = (alpha_a*cosBasis')*1;
    
    % design matrix method?
    % A = (convmtx((act),length(K_a)));
    % B = (convmtx((stim),length(K_s)));
    % F = A*K_a' + B*K_s' + base;
    act_h = act(1:end-1);
    stim_ = stim(2:end);
    act_ = act(2:end);
    mask_ = mask(2:end);
    F = conv_kernel(stim_',K_s) + conv_kernel(act_h',K_a) + base;
    F = F(1:length(act_));
    F(1:length(K_a)) = 0;
    F = F.*mask_;
    P = NL(F, M, m);

%%% log-likelihood
    y_1 = find(act_ == 1);
    y_0 = find(act_ == 0);
    epsilon = 1e-10;  % Small value to avoid log(0)
    P = max(min(P, 1 - epsilon), epsilon);
    ll = sum(log(P(:)).*act_) + sum((1-act_).*log(1-P(:))) - lambda*norm(K_a);
%     ll = sum(dth.*(log(P(y_logical))) + (1-dth).*log(1-P(y_logical)));
    NLL = -ll;
end

function [P] = NL(F,M,m)
    P = (M-m)./(1+exp(-F)) + m;
    % P = 1./(1+exp(-F));
end

function [F] = conv_kernel(X,K)
%%%
% using time series X and kernel K and compute convolved X with K
% this takes care of padding and the time delay
% note that K is a causal vector with zero-time in the first index
%%%
    padding = zeros(1,floor(length(K)/2));  %padding vector
    Fp = conv([padding, X, padding], K, 'same');  %convolvultion of vectors
    F = Fp(1:length(X));  %return the right size
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