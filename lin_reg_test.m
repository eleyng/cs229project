close all; clear all; clc; 

%[pawncdnum_steps, pawncdfeature_matrix, pawncdmetabolics] = extract_features('pawncd', 0); % train1


data_dir = './Data/PAWN'; 

%% PAWNED
%traindata = {'10_25/pawnfc',...
 %            '10_25/pawnfd',... % ,...
  %          '10_19/pawnec',...
   %         '10_19/pawned'}
            
             %'10_19/pawnee'...
             %'10_19/pawnef',...
             %'10_19/pawneh',...
             %'10_19/pawnei',...
             %'10_19/pawnel',...
             %'10_19/pawnem',...
             %'10_19/pawneo',...
             %'10_19/pawnep'};

traindata = {  '10_9/pawndc',...
               '10_9/pawndd',...
               '10_19/pawnec',...
               '10_19/pawned',...
               '10_25/pawnfc',...
               '10_25/pawnfd'
                }

traindata_sizes = zeros(length(traindata), 1); 



% ignoring opti trial from day 2 
%{
opti_trials = {'10_9/pawndc',...
               '10_9/pawndd',...
               '10_19/pawnec',...
               '10_19/pawned',...
               '10_25/pawnfc',...
               '10_25/pawnfd',...
               '11_4/pawngb',...
                }
%}


% should we group the ones that are from the same day? 

train_feature_matrix = [];      %x_data
train_metabolics = [];          %y_data


% good to know where day to day trials end 


for i = 1:numel(traindata)
    clear allData; % load(); 
    [feature_matrix, metabolics] = extract_features(fullfile(data_dir, traindata{i}), 2); % train2

    traindata_sizes(i) = length(metabolics); 


    %metabolics = metabolics - 83; 

    train_feature_matrix = [ train_feature_matrix; feature_matrix ] ;
    train_metabolics = [ train_metabolics; metabolics ];   
    
end


%% 
testdata = {'11_4/pawngb'} % ,...
             %'11_4/pawnfq',...
             %'11_4/pawnfr'}; 
[test_feature_matrix, test_metabolics] = extract_features(fullfile(data_dir, testdata{1}), 2); 

%test_metabolics = test_metabolics - 87.5;     % subtract average from that day 



%%

% Perform lin reg 
theta = train_feature_matrix\train_metabolics;

train_metabolics_predict = train_feature_matrix * theta;

% Calculate error
% lin_reg_err = 1/length(metabolics) * (metabolics - metabolics_predict).^2

%{
figure,
plot(train_metabolics, 'k'), hold on 
plot(train_metabolics_predict, 'b--')

% throw in some vertical delimeters 

% 10_9 delimeter 
plot(sum(traindata_sizes(1:2)) * ones(1,2), [-1e5, 1e5], 'r-'); 

% 10_19 delimeter 
plot(sum(traindata_sizes(1:4)) * ones(1,2), [-1e5, 1e5], 'r-'); 


title('Train')

xlim([0, sum(traindata_sizes)]); 
ylim([0, 300])

hold off 
%} 

figure, 
yl = [100, 320];
xl = [1, 36];


subplot(3, 1, 1)
plot(train_metabolics(1:sum(traindata_sizes(1:2))), 'k', 'linewidth', 2), hold on 
plot(train_metabolics_predict(1:sum(traindata_sizes(1:2))), 'b--', 'linewidth', 2)
ylim(yl)
xlim(xl)

subplot(3, 1, 2)
plot(train_metabolics((sum(traindata_sizes(1:2)) + 1):sum(traindata_sizes(1:4))), 'k', 'linewidth', 2), hold on 
plot(train_metabolics_predict((sum(traindata_sizes(1:2)) + 1):sum(traindata_sizes(1:4))), 'b--', 'linewidth', 2)
ylim(yl)
xlim(xl)
ylabel('Metabolic Cost')

subplot(3, 1, 3)
plot(train_metabolics((sum(traindata_sizes(1:4)) + 1):sum(traindata_sizes(1:6))), 'k', 'linewidth', 2), hold on 
plot(train_metabolics_predict((sum(traindata_sizes(1:4)) + 1):sum(traindata_sizes(1:6))), 'b--', 'linewidth', 2)
ylim(yl)
xlim(xl)
xlabel('Condition')

test_metabolics_predict = test_feature_matrix*theta; 
figure,
plot(test_metabolics, 'k', 'linewidth', 2), hold on 
plot(test_metabolics_predict, 'b--', 'linewidth', 2)
ylim(yl)
xlim(xl)


ylabel('Metabolic Cost')
xlabel('Condtion')


title('Test')



