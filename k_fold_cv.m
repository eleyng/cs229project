close all; clear all; clc; 


load('processed_data.mat');
eley_unit_var = data_scaling(eley_data);
michael_unit_var = data_scaling(michael_data); 


K = 5;

for person = 1:2

    if (person == 1)
        data_orig = michael_unit_var;
        metabolics_orig = michael_metabolics; 
        col = 'b'; 
    else
        bad_idx = find(eley_metabolics < 0); 
        eley_unit_var(bad_idx, :) = []; 
        eley_metabolics(bad_idx) = [];
        data_orig = eley_unit_var;
        metabolics_orig = eley_metabolics;
        col = 'r'; 
    end 



    np = length(data_orig); 
    perm = randperm(np); 
    data = data_orig(perm, :);     % randomly swap rows 
    metabolics = metabolics_orig(perm); 
    %% ----------------------- Linar Regression --------------------------
    % loglog(Lambda, kfold_lr_mse)
    % come up with index lists for K-fold sets 
    cv_sets = cell(K, 1);
    set_size = floor(np/K); 
    for i = 1:(K - 1)
        idx1 = set_size * (i - 1) + 1;
        idx2 = idx1 + set_size - 1;
        cv_sets{i} = idx1:idx2; 
    end 
    cv_sets{K} = (idx2 + 1):length(data); 
    full_indices = 1:length(data); 




      %% ----------------- Network cross validation + Lin Reg ------------
    data_no_control = data(:, 1:(end-4));
    data_step_only = data_no_control(:, 1:(end - 16)); 
    data_emg_only = data_no_control(:, (end - 15):end);

    [mse_list_all] = nn_kfold_br(data, metabolics, cv_sets);
    [mse_list_nc] = nn_kfold_br(data_no_control, metabolics, cv_sets);
    [mse_list_so] = nn_kfold_br(data_step_only, metabolics, cv_sets);
    [mse_list_emg] = nn_kfold_br(data_emg_only, metabolics, cv_sets);

    [mse_lr_all, ~] = k_fold_lr(data, metabolics, K);
    [mse_lr_nc, ~] = k_fold_lr(data_no_control, metabolics, K);
    [mse_lr_so, ~] = k_fold_lr(data_step_only, metabolics, K);
    [mse_lr_emg, ~] = k_fold_lr(data_emg_only, metabolics, K);


    fits.mse_list_all = mse_list_all;
    fits.mse_list_nc = mse_list_nc;
    fits.mse_list_so = mse_list_so;
    fits.mse_list_emg = mse_list_emg; 

    fits.mse_lr_all = mse_lr_all;
    fits.mse_lr_nc = mse_lr_nc;
    fits.mse_lr_so = mse_lr_so; 
    fits.mse_lr_emg = mse_lr_emg; 

    if (person == 1)
        michael_fits = fits; 
    else
        eley_fits = fits; 
    end 
end 

save('k_fold_data.mat');



function [best_mse, R_lr] = k_fold_lr(data, metabolics, K);

        % Bias term will already be added 
    Lambda = logspace(-5,-1,25);

    Mdl = fitrlinear(data', metabolics, 'ObservationsIn', 'columns',...
                                 'KFold', K, 'Lambda', Lambda,...
                         'Learner', 'leastsquares', 'Regularization', 'lasso');   % because we took transpose for speed 
    kfold_lr_mse = kfoldLoss(Mdl);
    [best_mse, idx] = min(kfold_lr_mse); 
    


    % MdlFinal = selectModels(Mdl, idx); 

    lin_reg_pred_all = kfoldPredict(Mdl); 
    best_pred = lin_reg_pred_all(:, idx); 

    correlation_coeff = corr2(best_pred, metabolics);
    r_sqr_lr = power(correlation_coeff,2);
    R_lr = sqrt(r_sqr_lr); 
    display(R_lr); 
end 



function [neuron_mse_list] = nn_kfold_br(data, metabolics, cv_sets)

    K = length(cv_sets); 
    full_indices = 1:length(metabolics);

    max_neurons = 14; 
    neuron_mse_list = zeros(1, max_neurons); 

    data_try = data';
    targets = metabolics';

    for n = 1:max_neurons
        hiddenLayerSize = n; 
        cv_mse = zeros(1, K); 
        fprintf('K-fold trying %d neuron(s):', n); 
        for i = 1:K     % K-fold 

            net = feedforwardnet(hiddenLayerSize, 'trainbr');    % bayesian regulaurization 

            trainInd = setdiff(full_indices, cv_sets{i});
            testInd = cv_sets{i}; 



            % Set up Division of Data for Training, Validation, Testing
            net.divideFcn = 'divideind'; 
            net.divideParam.trainInd = trainInd;
            net.divideParam.testInd = testInd;
            %net.divideParam.testRatio = testRatio;

            net.trainParam.epochs = 500; 
            net.trainParam.goal = 0.5e-5;     % maybe even high because met is noisey 

            % Train the Network
            net.trainParam.showWindow = false;
            [trained_net,tr] = train(net,data_try,targets);

            % Test the Network
            outputs = trained_net(data_try);
            errors = gsubtract(outputs,targets);
            performance = perform(trained_net,targets,outputs); %MSE test

            %avg_mse(trial) = performance;
            cv_mse(i) = performance;
        end 

        neuron_mse_list(n) = mean(cv_mse); 
        fprintf(' Mean mse: %0.5f\n', neuron_mse_list(n)); 
    end 


end 

