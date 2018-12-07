%% Load dataset by running data_processing.m
inputs = data;
targets = full_metabolics';

%% CrossValidation with number of features
% Using best subset

K = 5; % number of folds
[m, n] = size(inputs); % m samples, n predictors
numCVtrials = 20; % number of trials for cross validation
trainRatio = 70 / 100;
valRatio = 15 / 100;
testRatio = 1 - (trainRatio + valRatio);

% Percent train
Ptr = 0.7;
Pval = 1 - Ptr;
%Pval = 0.15;
%Pte = 1 - (Ptr + Pval);

% Constants for neural network architecture
hiddenLayerSize = 10; %1 for linear regression
trainFcn = 'trainlm';

data_cpy = inputs;
it = 0; % iterations/counter
overall_models_tried = zeros(n, n); % matrix of best sub models tried encoded, where each bool represents predictor is in use or not
overall_models_mse = zeros(n, 1); % always just store the min after each submodel 

sub_model = []; %building submodel
row = zeros(1, n); %overall model encoding
ind = 1:n; %keep track; remove idx after selecting best submodel

% LOOP: Forward stepwise model selection
while it < n
    
    %LOOP: Try a submodel by creating sub-dataset
    sub_model_mse = zeros(size(n-it, 1));
    
    i = 1;
    while i <= size(ind, 2)
        
        test_model = [sub_model ind(i)];
        data_try = inputs(:, test_model)';
      
        % LOOP: number of CV trials
        avg_mse = zeros(size(numCVtrials, 1));
        
        for trial=1:numCVtrials

%             % Randomly split dataset (is this needed?)
%             idx = randperm(m);
%             train = data_try( idx(1:round(P*m)), : );
%             val = data_try( idx(round(P*m)+1 : end), : );
            %train = data_try';

            % Train the network with training split
            % Create a network

            net = feedforwardnet(hiddenLayerSize, trainFcn);

            % Set up Division of Data for Training, Validation, Testing
            net.divideParam.trainRatio = trainRatio;
            net.divideParam.valRatio = valRatio;
            net.divideParam.testRatio = testRatio;

            % Train the Network
            net.trainParam.showWindow = false;
            [net,tr] = train(net,data_try,targets);

            % Test the Network
            outputs = net(data_try);
            errors = gsubtract(outputs,targets);
            performance = perform(net,targets,outputs); %MSE test
            
            avg_mse(trial) = performance;
            
        end
        
        avg_mse = mean(avg_mse);
        sub_model_mse(i) = avg_mse;
        i = i + 1;
    end
    
    [best_sub_mse, best_sub_model] = min(sub_model_mse); %find the best submodel
    sub_model = [sub_model ind(best_sub_model)];
    
    row = zeros(n, 1); row(sub_model(it + 1)) = 1; %mark the sub model tried
    overall_models_tried(it+1, :) = row; % add to the matrix of best sub models tried
    overall_models_mse(it+1) = best_sub_mse; % always just store the min after each submodel 
    
    ind(best_sub_model) = []; %remove predictor that's been selected for best submodel
    it = it + 1;
    
end

% Find overall best model
[overall_best_sub_mse, overall_best_sub_model] = min(overall_models_mse); %find the best submodel
best_model = sum(overall_models_tried(1:overall_best_sub_model, :)); % best model
find(best_model == 0) % find the predictors do not contribute

%% CrossValidation with hidden layer size
% TODO if time permits
