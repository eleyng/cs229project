% Plots + Tables for Paper + Poster 
% CS229 Project, Fall 2018


close all; clear all; clc; 




%%----------------------- K-Fold Plots -----------------------
load('k_fold_data.mat'); 


ylims = [2, 20] * 1e-3; 

% All features 
figure, hold on 
plot(michael_fits.mse_list_all, 'bo-', 'DisplayName', 'S1 All Features');
plot([1, length(michael_fits.mse_list_all)], min(michael_fits.mse_lr_all) * ones(1, 2), 'b--', 'DisplayName', 'S1 Linear Regression')
plot(eley_fits.mse_list_all, 'ro-', 'DisplayName', 'S2 All Features');
plot([1, length(eley_fits.mse_list_all)], min(eley_fits.mse_lr_all) * ones(1, 2), 'r--', 'DisplayName', 'S2 Linear Regression')
legend show
xlabel('Number of Neurons');
ylabel('MSE')
ylim(ylims);
hold off 
print('cv_all_features', '-dpng', '-r300'); 

% No Control
figure, hold on 
plot(michael_fits.mse_list_nc, 'bo-', 'DisplayName', 'S1 Step + EMG Data');
plot([1, length(michael_fits.mse_list_nc)], min(michael_fits.mse_lr_nc) * ones(1, 2), 'b--', 'DisplayName', 'S1 Linear Regression')
plot(eley_fits.mse_list_nc, 'ro-', 'DisplayName', 'S2 Step + EMG Data');
plot([1, length(eley_fits.mse_list_nc)], min(eley_fits.mse_lr_nc) * ones(1, 2), 'r--', 'DisplayName', 'S2 Linear Regression')
legend show
xlabel('Number of Neurons');
ylabel('MSE')
ylim(ylims);
hold off 
print('cv_no_control', '-dpng', '-r300'); 


% Step Only
figure, hold on 
plot(michael_fits.mse_list_so, 'bo-', 'DisplayName', 'S1 Step Only');
plot([1, length(michael_fits.mse_list_so)], min(michael_fits.mse_lr_so) * ones(1, 2), 'b--', 'DisplayName', 'S1 Linear Regression')
plot(eley_fits.mse_list_so, 'ro-', 'DisplayName', 'S2 Step Only');
plot([1, length(eley_fits.mse_list_so)], min(eley_fits.mse_lr_so) * ones(1, 2), 'r--', 'DisplayName', 'S2 Linear Regression')
legend show
xlabel('Number of Neurons');
ylabel('MSE')
ylim(ylims);
hold off 
print('cv_step_only', '-dpng', '-r300'); 

% Emg Only 
figure, hold on 
plot(michael_fits.mse_list_emg, 'bo-', 'DisplayName', 'S1 EMG Only');
plot([1, length(michael_fits.mse_list_emg)], min(michael_fits.mse_lr_emg) * ones(1, 2), 'b--', 'DisplayName', 'S1 Linear Regression')
plot(eley_fits.mse_list_emg, 'ro-', 'DisplayName', 'S2 EMG Only');
plot([1, length(eley_fits.mse_list_emg)], min(eley_fits.mse_lr_emg) * ones(1, 2), 'r--', 'DisplayName', 'S2 Linear Regression')
legend show
xlabel('Number of Neurons');
ylabel('MSE')
ylim(ylims);
hold off 
print('cv_emg_only', '-dpng', '-r300'); 

%    set(0, 'currentfigure', fig); hold on
 %   plot(mse_list_all, [col, 'x-'], 'DisplayName', 'All Parameters');
  %  plot(mse_list_nc, [col, 'o-'], 'DisplayName', 'Step + EMG');
  %  plot(mse_list_so, [col, 'd-'], 'DisplayName', 'Step Data Only');
   % plot(mse_list_emg, [col, 's-'], 'DisplayName', 'EMG Only');
    %plot([1, length(mse_list_all)], min(kfold_lr_mse) * ones(1, 2), [col, '--'], 'DisplayName', 'Linear Regression')
    %hold off  
%set(0, 'currentfigure', fig); legend(show); 



%%




load('processed_data.mat'); 
eley_unit_var = data_scaling(michael_data);
michael_unit_var = data_scaling(michael_data); 








%% Make Data Zero Mean and Unit Variance 




%perm = randperm(np_e); 
%eley_permute = eley_unit_var(perm, :);     % randomly swap rows 
%eley_met_lr = eley_metabolics(perm); 
%eley_lr = [eley_permute, ones(length(eley_permute), 1)]; 





% lr_model = CVMdl.Trained

%theta_best = 
%}

%%---------------------------- PCA --------------------------------------


% Take all the data first and make it unit mean and variance 



%[coeff_all, score_all, latent_all, tsquared_all, explained_all, mu_all] = pca(all_unit_var);
%U_all = coeff_all(:, 1:num_pcs_all);
%pca_all = all_unit_var * U_all; 


% Eley Data 
%num_pcs_all = 25;
%num_pcs_nc = 25;

eley_no_controls = eley_unit_var(:, 1:(end - 4));
eley_emg_only = eley_unit_var(:, 10:25); 
eley_step_only = eley_unit_var(:, 1:9); 

% All data 
[coeff_e_all, score_e_all, latent_e_all, tsquared_e_all, explained_e_all, mu_e_all] = pca(eley_unit_var, 'NumComponents', 23);
%U_e_all = coeff_e_all(:, 1:num_pcs_all);
pca_e_all = coeff_e_all * score_e_all'

[coeff_e_nc, score_e_nc, latent_e_nc, tsquared_e_nc, explained_e_nc, mu_e_no] = pca(eley_no_controls);
%U_e_nc = coeff_e_nc(:, 1:num_pcs_nc);
pca_e_nc = coeff_e_nc * score_e_nc'; 


[coeff_e_emg, score_e_emg, latent_e_emg, tsquared_e_emg, explained_e_emg, mu_e_emg] = pca(eley_emg_only);
[coeff_e_so, score_e_so, latent_e_so, tsquared_e_so, explained_e_so, mu_e_so] = pca(eley_step_only);

data = pca_e_nc;
metabolics = data_scaling(michael_metabolics);

inputs = data';
targets = metabolics';

[m, n] = size(inputs); % m samples, n predictors
numCVtrials = 5; % number of trials for cross validation
trainRatio = 65 / 100;
valRatio = 0 / 100;
testRatio = 1 - (trainRatio + valRatio);

% Parameters for neural network architecture
hiddenLayerSize = 4; %1 for linear regression
trainFcn = 'trainbr';


%% WRONG METHOD: IGNORE
% Cannot make test and train PCA because those will be different PCs
numTrials = 100;

for idx=1:numTrials

    net = feedforwardnet(hiddenLayerSize, trainFcn);

    % Set up Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.testRatio = testRatio;

    % Train the Network
    net.trainParam.showWindow = false;
    [net,~] = train(net,inputs,targets);

    % Test the Network
    outputs = net(inputs);
    %errors = gsubtract(outputs,targets);
    performance = perform(net,targets,outputs); %MSE test

    %avg_mse(trial) = performance;
    mean_mse_lst(idx) = performance;
    
end

mean_mse = mean(mean_mse_lst)
%%

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
