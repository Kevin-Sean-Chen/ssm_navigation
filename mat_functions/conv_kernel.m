function [F] = conv_kernel(X,K)
%%%
% using time series X and kernel K and compute convolved X with K
% this takes care of padding and the time delay
% note that K is a causal vector with zero-time in the first index
%%%
    % pad_l = floor(length(K)/2);
    % padding = zeros(1,pad_l) + mean(X);  %padding vector
    % Fp = conv([padding, X, padding], K, 'same');  %convolvultion of vectors
    % F = Fp(pad_l+1:length(X)+pad_l);  %return the right size
    
    %%% old code
    % pad_l = floor(length(K)/2);
    % padding = zeros(1,pad_l);% + mean(X);  %padding vector
    % Fp = conv([padding, X, padding], K, 'same');  %convolvultion of vectors
    % F = Fp(1:length(X));  %return the right size

    %%% trust matlab
    Fp = conv(X, K, 'full');
    F = Fp(1:length(X));
end