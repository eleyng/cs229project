
close all; clear all; clc; 
[feature_matrix, metabolics] = extract_features('pawnci'); 


% Perform lin reg 
theta = feature_matrix\metabolics; 

metabolics_predict = feature_matrix * theta;

figure,
plot(metabolics, 'k'), hold on 
plot(metabolics_predict, 'b--')
hold off 


