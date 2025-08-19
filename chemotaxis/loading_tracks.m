% bacterial_states
clear
clc

%% A stand-alone script to investigate behavioral states in bacteria
% #########################################################################
% load data from .mat file
% plot tracks and compute basic stats
% build feature vectors for clustering
% if cluster works, build transition models
% sample from the model and see if it matches with data
% #########################################################################
%% load mat file for data
load('D:\github\behavior_state_space\data\PAK_1.rad_swimtracker.mat')

%% basic variables
n_tracks = length(tracks);

ith_cell = 1;
figure()
plot(tracks(ith_cell).x, tracks(ith_cell).y)

%% scatter plot
figure()
for ii = 1:500
    tracki = tracks(ii);
    plot(tracki.speed, tracki.angvelocity ,'k.'); hold on;
end

%% average autcorrelation
% this gives a sense for the time scales
nlags = 100;
population_acf = [];
for ii = 1:1000
    [aa,bb] = autocorr(tracks(ii).speed, 'NumLags',nlags);
    % [aa,bb] = autocorr(tracks(ii).angvelocity, 'NumLags',nlags);
    if length(aa)>nlags
        population_acf = [population_acf aa(1:nlags)];
    end
end
figure()
plot(mean(population_acf,2))