close all; clear all; clc; 
startup
%[pawncdnum_steps, pawncdfeature_matrix, pawncdmetabolics] = extract_features('pawncd', 0); % train1

%% PAWNED
traindata = ["pawnec", "pawned"];
train_feature_matrix = []; %x_data
train_metabolics = []; %y_data

for filename = traindata
    clear allData; load(filename)
    [feature_matrix, metabolics] = extract_features(filename, 2); % train2
    train_feature_matrix = [ train_feature_matrix; feature_matrix ] ;
    train_metabolics = [ train_metabolics; metabolics ];   
    
end

%%

% Perform lin reg 
theta = feature_matrix\metabolics;

metabolics_predict = feature_matrix * theta;

% Calculate error
lin_reg_err = 1/length(metabolics) * (metabolics - metabolics_predict).^2

figure,
plot(metabolics, 'k'), hold on 
plot(metabolics_predict, 'b--')
hold off 