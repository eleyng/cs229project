%data = data_scaling(michael_data(:, 1:25));
metabolics = data_scaling(michael_metabolics);

% After running feature selection
features = [4,20,3,16,10,14,19,8,21,23,13,25,22,9,6,17,11,5,24,7,15,2,12];
data = data_scaling(michael_data(:, features));

inputs = data';
targets = metabolics';

[m, n] = size(inputs); % m samples, n predictors
numCVtrials = 10; % number of trials for cross validation
trainRatio = 70 / 100;
testRatio = 1 - (trainRatio);

% Parameters for neural network architecture
hiddenLayerSize = 20; %1 for linear regression
trainFcn = 'trainbr';

numTrials = 1;

for idx=1:numTrials

    net = feedforwardnet(hiddenLayerSize, trainFcn);

    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.testRatio = testRatio;

    % Train the Network
    net.trainParam.showWindow = true;
    net.trainParam.epochs = 500; 
    net.trainParam.goal = 0.5e-5; 
    [net,~] = train(net,inputs,targets);

    % Test the Network
    outputs = net(inputs);
    %errors = gsubtract(outputs,targets);
    performance = perform(net,targets,outputs); %MSE test

    %avg_mse(trial) = performance;
    mean_mse_lst(idx) = performance;
    
end
%%
mean_mse = mean(mean_mse_lst)