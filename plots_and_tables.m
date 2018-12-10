% Plots + Tables for Paper + Poster 
% CS229 Project, Fall 2018


close all; clear all; clc; 
load('processed_data.mat'); 








%% Make Data Zero Mean and Unit Variance 

eley_unit_var = data_scaling(eley_data);
michael_unit_var = data_scaling(michael_data); 




%perm = randperm(np_e); 
%eley_permute = eley_unit_var(perm, :);     % randomly swap rows 
%eley_met_lr = eley_metabolics(perm); 
%eley_lr = [eley_permute, ones(length(eley_permute), 1)]; 





K = 5;

fig = figure; 


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


    % Bias term will already be added 

    Lambda = logspace(-5,-1,25);
    Mdl = fitrlinear(data', metabolics, 'ObservationsIn', 'columns',...
                                 'KFold', K, 'Lambda', Lambda, 'Learner', 'leastsquares', 'Regularization', 'lasso');   % because we took transpose for speed 
    kfold_lr_mse = kfoldLoss(Mdl);
    [best_lambda, idx] = min(kfold_lr_mse); 
    % MdlFinal = selectModels(Mdl, idx); 

    lin_reg_pred_all = kfoldPredict(Mdl); 
    best_pred = lin_reg_pred_all(:, idx); 

    correlation_coeff = corr2(best_pred, metabolics);
    r_sqr_lr = power(correlation_coeff,2);
    R_lr = sqrt(r_sqr_lr); 
    display(R_lr); 

    %X = X'; 
    %CVMdl = fitrlinear(X,Y,'ObservationsIn','columns','KFold',5,'Lambda',Lambda,...
    %    'Learner','leastsquares','Solver','sparsa','Regularization','lasso');

    %numCLModels = numel(CVMdl.Trained)

    % loglog(Lambda, kfold_lr_mse)
w

    %% ----------------- Network cross validation ---------------------------------------


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



    
    data_no_control = data(:, 1:(end-4));
    data_step_only = data_no_control(:, 1:(end - 16)); 
    data_emg_only = data_no_control(:, (end - 15):end);

    [mse_list_all] = nn_kfold_br(data, metabolics, cv_sets);
    [mse_list_nc] = nn_kfold_br(data_no_control, metabolics, cv_sets);
    [mse_list_so] = nn_kfold_br(data_step_only, metabolics, cv_sets);
    [mse_list_emg] = nn_kfold_br(data_emg_only, metabolics, cv_sets);

    set(0, 'currentfigure', fig); hold on
    plot(mse_list_all, [col, 'x-'], 'DisplayName', 'All Parameters');
    plot(mse_list_nc, [col, 'o-'], 'DisplayName', 'Step + EMG');
    plot(mse_list_so, [col, 'd-'], 'DisplayName', 'Step Data Only');
    plot(mse_list_emg, [col, 's-'], 'DisplayName', 'EMG Only');
    plot([1, length(mse_list_all)], min(kfold_lr_mse) * ones(1, 2), [col, '--'], 'DisplayName', 'Linear Regression')
    hold off  

end 

set(0, 'currentfigure', fig); legend(show); 
% lr_model = CVMdl.Trained

%theta_best = 
%}

%%---------------------------- PCA --------------------------------------


% Take all the data first and make it unit mean and variance 



%[coeff_all, score_all, latent_all, tsquared_all, explained_all, mu_all] = pca(all_unit_var);
%U_all = coeff_all(:, 1:num_pcs_all);
%pca_all = all_unit_var * U_all; 


% Eley Data 
num_pcs_all = 8;
num_pcs_nc = 8;

eley_no_controls = eley_unit_var(:, 1:(end - 4));
eley_emg_only = eley_unit_var(:, 10:25); 
eley_step_only = eley_unit_var(:, 1:9); 

% All data 
[coeff_e_all, score_e_all, latent_e_all, tsquared_e_all, explained_e_all, mu_e_all] = pca(eley_unit_var);
U_e_all = coeff_e_all(:, 1:num_pcs_all);
pca_e_all = eley_unit_var * U_e_all; 

[coeff_e_nc, score_e_nc, latent_e_nc, tsquared_e_nc, explained_e_nc, mu_e_no] = pca(eley_no_controls);
U_e_nc = coeff_e_nc(:, 1:num_pcs_nc);
pca_e_nc = eley_no_controls * U_e_nc; 


[coeff_e_emg, score_e_emg, latent_e_emg, tsquared_e_emg, explained_e_emg, mu_e_emg] = pca(eley_emg_only);
[coeff_e_so, score_e_so, latent_e_so, tsquared_e_so, explained_e_so, mu_e_so] = pca(eley_step_only);



figure, hold all
plot(cumsum(explained_e_all), 'DisplayName', 'All Features')
plot(cumsum(explained_e_nc), 'DisplayName', 'Step + EMG')
plot(cumsum(explained_e_emg), 'DisplayName', 'EMG Only')
plot(cumsum(explained_e_so), 'DisplayName', 'Step Only')
legend show; 
hold off 






%U_all = coeff_all(:, 1:num_pcs_all);
%pca_all = all_unit_var * U_all; 


% Michael Data 



%{

data_no_emg = data(:, 1:(end-16)); 
emg_only = data(:, (end-15):end);   % for PCA, woot woot!

left_emg_idx = [1:4, 9:12]; 
right_emg_idx = [5:8, 13:16]; 




%% pca on the emg becase "machine learning"

% take all the emg make zero mean and unit variance
emg_for_pca = bsxfun(@minus, emg_only, mean(emg_only));
emg_for_pca = bsxfun(@rdivide, emg_for_pca, sqrt(var(emg_for_pca)));
[coeff,score,latent, tsquared, explained, mu] = pca(emg_for_pca);

num_pcs = 6; 
U = coeff(:, 1:num_pcs);
pca_emg = emg_for_pca * U; 



% find mean RMS for constant prediction 

constant_pred_rms = rms(full_metabolics - mean(full_metabolics));
constant_pred_mse = constant_pred_rms^2; 
display(constant_pred_rms); 
display(constant_pred_mse); 

%}


% 


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