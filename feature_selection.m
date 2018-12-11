%% CrossValidation with number of features
% Using forward stepwise selection

%% Load dataset by running data_processing.m
data = data_scaling(michael_data(:, 1:25));
metabolics = data_scaling(michael_metabolics);

% After running feature selection
%features = [4,20,3,16,10,14,19,8,21,23,13,25,22,9,6,17,11,5,24,7,15,2,12];
%data = data_scaling(michael_data(:, features));

inputs = data;
targets = metabolics';

% Parameters for feature selection 
numTrials = 5;
[m, n] = size(inputs);
models_lst = zeros(numTrials, n);
mse_lst = zeros(numTrials, 1);
build_sub_model = zeros(numTrials, n);
build_sub_model_best_mse = zeros(numTrials, n);
indexing = 1:n;


% Run feature selection
for tr=1:numTrials
    %[overall_best_model, overall_best_sub_mse] = feature_selection(inputs, targets)
    
    %K = 5; % number of folds
    [m, n] = size(inputs); % m samples, n predictors
    numCVtrials = 5; % number of trials for cross validation
    trainRatio = 70 / 100;
    testRatio = 1 - (trainRatio);

    % Parameters for neural network architecture
    hiddenLayerSize = 2; %1 for linear regression
    trainFcn = 'trainbr';

    it = 0; % iterations/counter
    overall_models_tried = zeros(1, n); % matrix of best sub models tried encoded, where each bool represents predictor is in use or not
    overall_models_mse = zeros(1, n); % always just store the min after each submodel
    model_build = zeros(1, n); % see the order in which it was built

    sub_model = []; %building submodel
    ind = 1:n; %keep track; remove idx after selecting best submodel

    % LOOP: Forward stepwise model selection
    while it < n

        %LOOP: Try a submodel by creating sub-dataset
        sub_model_mse = zeros(n-it,1);

        i = 1;
        while i <= size(ind, 2)

            test_model = [sub_model, ind(i)];
            data_try = inputs(:, test_model)';

            % LOOP: number of CV trials
            %avg_mse = zeros(size(numCVtrials, 1));
            mean_mse_lst = zeros(numCVtrials,1);

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
                net.divideParam.testRatio = testRatio;

                % Train the Network
                net.trainParam.showWindow = false;
                [net,~] = train(net,data_try,targets);

                % Test the Network
                outputs = net(data_try);
                %errors = gsubtract(outputs,targets);
                performance = perform(net,targets,outputs); %MSE test

                %avg_mse(trial) = performance;
                mean_mse_lst(trial) = performance;

            end

            %avg_mse = mean(avg_mse);
            %sub_model_mse(i) = avg_mse;
            mean_mse = median(mean_mse_lst);
            sub_model_mse(i) = mean_mse;
            i = i + 1;
        end

        [best_sub_mse, best_sub_model_idx] = min(sub_model_mse); %find the index of best submodel
        
        % keep track of how the model is being built
        sub_model(it + 1) = ind(best_sub_model_idx);
        sub_model_best_mse(it + 1) = best_sub_mse;

        %row = zeros(1, n); row(sub_model(it + 1)) = 1; %mark the sub model tried
        overall_models_tried(it + 1) = ind(best_sub_model_idx); % add to the feature to the list of tried models
        overall_models_mse(it + 1) = best_sub_mse; % always just store the min after each submodel

        ind(best_sub_model_idx) = []; %remove predictor that's been selected for best submodel

        if it < 10
            hiddenLayerSize = hiddenLayerSize + 1;
        else
            hiddenLayerSize = 10 ;
        end

        it = it + 1;

    end

    % Find overall best model
    [overall_best_sub_mse, overall_best_sub_model_idx] = min(overall_models_mse); %find the best submodel
    overall_best_model = overall_models_tried(1:overall_best_sub_model_idx); % best model
    
    models_lst(tr, overall_best_model) = overall_best_model;
    mse_lst(tr) = overall_best_sub_mse;
    
    build_sub_model(tr, :) = overall_models_tried;
    build_sub_model_best_mse(tr, :) = overall_models_mse;
    
    %overall_sub_model = [overall_sub_model; sub_model];
    %overall_sub_model_best_mse = [overall_sub_model_best_mse; sub_model_best_mse];
    figure;
    plot(overall_models_mse); title('MSE vs. Feature Added'); xlabel('ith Feature Added'); ylabel('MSE');
    
    hold off;
end


%% CrossValidation with hidden layer size
% TODO if time permits

% test1 : michael's data with controls, hiddenlayersize
% test2 : michael's data without controls, hiddenlayersize up to 20