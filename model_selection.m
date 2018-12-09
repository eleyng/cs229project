%% Load dataset by running data_processing.m
inputs = data;
targets = full_metabolics';

fctioileID = fopen('model_selection_output.txt','w');

% Parameters for feature selection 
numTrials = 10;
[m, n] = size(inputs);
dummyvec = zeros(1, n);

% Run feature selection
for tr=1:numTrials
    model = feature_selection(inputs, targets)
    model(numel(dummyvec)) = 0;
    best_models_over_trials(tr, :) = model;
end

best_models_over_trials

