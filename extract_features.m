function [features, output] = extract_features(filename, avg)

% Load Contents from  data file into function workspace 
load(filename)


firstIdx = 60000;    % ignore first two minutes of data 

% Run evalvars script on function workspace 
if length(allData.signalNames) > 50
names = {'posaL','posaR','velaL','velaR','posmL','posmR','velmL','velmR',...
    'tauL','tauR','hsL','hsR',...
    'LFx','LFy','LFz','LMx','LMy','LMz',...
    'RFx','RFy','RFz','RMx','RMy','RMz',...
    'posmdesL','velmdesL','taudesL','posmdesR','velmdesR','taudesR',...
    'phaseL','tstrideL','tstanceL','ctrlL','walkL',...
    'phaseR','tstrideR','tstanceR','ctrlR','walkR',...
    'vo2','vco2','cmaesenable','cost','condNo','met',...
    'cmap1','cmap2','cmap3','cmap4',...'time'};%,...
    'aL','aR','qL','qR',...
    'vicon','time'};
%     ...'a1','a2','a3','a4','q1','q2','q3','q4','time'};
elseif length(allData.signalNames) > 19
    names = {'posaL','posaR','velaL','velaR','posmL','posmR','velmL','velmR',...
    'tauL','tauR','hsL','hsR',...
    'posmdesL','velmdesL','taudesL','posmdesR','velmdesR','taudesR',...
    'phaseL','tstrideL','tstanceL','ctrlL','walkL',...
    'phaseR','tstrideR','tstanceR','ctrlR','walkR',...
    'vo2','vco2','cmaesenable','cost','condNo','met',...
    'cmap1','cmap2','cmap3','cmap4',...'time'};%,...
    'aL','aR','qL','qR','time'};
else
names = {'aL','aR','qL','qR','posmL','posmR',...
    'turnsL','turnsR','adegL','adegR','qdegL','qdegR',...
    'aLvoltOff', 'aRvoltOff', 'qLvoltOff', 'qRvoltOff','timem'};
end

for i=1:length(names)
    eval([names{i} ' = allData.data(firstIdx:end,i);']);
end


dt = time(2) - time(1); 


% Metabolics filtering, may want to remove 
%pts = 3000;
%b = (1/pts) * ones(1, pts); 
%met = filter(b, 1, met);    

% NOTE -- may want to not do any fitting to first minute or so of test 





%% Cutting out first minute of data 


% Center of pressure 
v_tm = 1.25;    % treadmill speed, m/s 

h = 15e-3; % heigh of treadmill plane above sensor belt
L_xp = (-h * LFx - LMy)./LFz; 
L_yp = (h * LFy + LMx)./LFz; 

R_xp = (-h * RFx - RMy)./RFz; 
R_yp = (h * RFy + RMx)./RFz; 

dt = time(2) - time(1);

LFz = LFz - min(LFz);
RFz = RFz - min(RFz);

LFz_bool = ceil(LFz - .05); % was 0.2 
LFz_bool(LFz_bool > 0) = 1; 


L_toe_off_idx = find(diff(LFz_bool) == -1);

% work with this for now -- clean up later 
num_steps = length(L_toe_off_idx) - 2;  % ignore first and last step  

% features initialization
metabolics = zeros(num_steps, 1); 

LFz_max = zeros(num_steps, 1);
LFz_max_idx = zeros(num_steps, 1); 
LFz_ttp =  zeros(num_steps, 1);      % time to peak 

L_stride_length = zeros(num_steps, 1); 


RFz_max = zeros(num_steps, 1); 
RFz_max_idx = zeros(num_steps, 1);
RFz_ttp =  zeros(num_steps, 1);      % time to peak 

R_stride_length = zeros(num_steps, 1);

% look at stride length from toe off to toe off 

L_max_plantar = zeros(num_steps, 1);
L_max_dorsi = zeros(num_steps, 1); 

R_max_plantar = zeros(num_steps, 1);
R_max_dorsi = zeros(num_steps, 1); 

cparam1 = zeros(num_steps, 1);
cparam2 = zeros(num_steps, 1);
cparam3 = zeros(num_steps, 1);
cparam4 = zeros(num_steps, 1);

% skip first step 


for i = 1:num_steps

    indices = L_toe_off_idx((i + 1):(i + 2));   % ignore first step 
    idx_start = indices(1);
    idx_end = indices(2) - 1; 
    idx_diff = idx_end - idx_start; 

    %% Peak toe off and timing 
    [LF_max, LF_max_idx] = max(LFz((idx_start + round(.75 * idx_diff)):idx_end));
    [RF_max, RF_max_idx] = max(RFz((idx_start + round(0.25 *idx_diff)):(idx_start + round(0.5 * idx_diff))));

    LFz_max(i) = LF_max;
    LFz_max_idx(i) = idx_start + round(.75 * idx_diff) + LF_max_idx - 1; 
    LFz_ttp(i) = dt *  (LFz_max_idx(i) - idx_start);  

    RFz_max(i) = RF_max;
    RFz_max_idx(i) = idx_start + round(.25 * idx_diff) + RF_max_idx - 1; 
    RFz_ttp(i) = dt *  (RFz_max_idx(i) - idx_start);  

    %% Stride Length -- calcultate from previous toe off to this toe off 
    if (i == 1)     % need to calculate previous center off pressure 
        idx_start_prev =  L_toe_off_idx(1);
        idx_end_prev = L_toe_off_idx(2) - 1; 
        idx_diff_prev = idx_end_prev - idx_start_prev; 

        %% Peak toe off and timing 
        [LF_max_prev, LF_max_idx_prev] = max(LFz((idx_start_prev + round(.75 * idx_diff_prev)):idx_end_prev));
        [RF_max_prev, RF_max_idx_prev] = max(RFz((idx_start_prev + ...
                                    round(0.25 *idx_diff_prev)):(idx_start_prev + round(0.5 * idx_diff_prev))));
        LFz_max_idx_prev = idx_start + round(.75 * idx_diff_prev) + LF_max_idx_prev - 1; 
        RFz_max_idx_prev = idx_start + round(.25 * idx_diff_prev) + RF_max_idx_prev - 1; 

       
    else 
        LFz_max_idx_prev = LFz_max_idx(i - 1);
        RFz_max_idx_prev = RFz_max_idx(i - 1);  
    end 

    prev_cop_Lx = L_xp(LFz_max_idx_prev);
    prev_cop_Ly = L_yp(LFz_max_idx_prev);

    prev_cop_Rx = R_xp(RFz_max_idx_prev);
    prev_cop_Ry = R_yp(RFz_max_idx_prev); 

    % Find current center off pressure to calculate stride
    cur_cop_Lx = L_xp(LFz_max_idx(i));
    cur_cop_Ly = L_yp(LFz_max_idx(i));

    cur_cop_Rx = R_xp(RFz_max_idx(i));
    cur_cop_Ry = R_yp(RFz_max_idx(i)); 

    % need time different for left and right 
    deltaT_L = dt * (LFz_max_idx(i) - LFz_max_idx_prev);
    deltaT_R = dt * (RFz_max_idx(i) - RFz_max_idx_prev); 


    % TODO Think about adding in x-movement of center of pressure as a feature


    % Y is along walking direction 
    L_stride_length(i) = (v_tm * deltaT_L) + (cur_cop_Ly - prev_cop_Ly); 
    R_stride_length(i) = (v_tm * deltaT_R) + (cur_cop_Ry - prev_cop_Ry); 

    %% Max Plantarflexion and Dorsiflexion angles 
    L_max_plantar(i) =  max(posaL(idx_start:idx_end)); 
    L_max_dorsi(i) = min(posaL(idx_start:idx_end)); 

    R_max_plantar(i) = max(posaR(idx_start:idx_end));
    R_max_dorsi(i) = min(posaR(idx_start:idx_end)); 
    
    %% Control Parameters
    cparam1(i) = mean(cmap1(idx_start:idx_end));
    cparam2(i) = mean(cmap2(idx_start:idx_end));
    cparam3(i) = mean(cmap3(idx_start:idx_end));
    cparam4(i) = mean(cmap4(idx_start:idx_end));

    %% Index the per-step metabolics average 
    metabolics(i) = mean(met(idx_start:idx_end)); 
    
    
    
    
end 
   %% Some useful code for debuggign if feature extraction is correct 

figure, plot(time, LFz, 'k'), hold on 
plot(time, RFz, 'g')
plot(time(L_toe_off_idx), LFz(L_toe_off_idx), 'rx')

plot(time(LFz_max_idx), LFz_max, 'ko');
plot(time(RFz_max_idx), RFz_max, 'go');

%plot(time, posaL, 'k--')
%plot(time, posaR, 'g--' )
hold off 


figure, plot(LFz_max_idx, L_stride_length, 'k'), hold on  
plot(RFz_max_idx, R_stride_length, 'g')
hold off 


%%


% come up with feature vector for every step --


feature_matrix = [LFz_max, RFz_max, LFz_ttp, RFz_ttp, L_stride_length, R_stride_length, L_max_dorsi,...
                                 R_max_dorsi, L_max_plantar, R_max_plantar, cparam1, cparam2, cparam3, cparam4, ones(num_steps, 1)];
                           


% Assign outputs 
features = feature_matrix; 
output = metabolics;
num_steps = 1:num_steps;


% Calc number of features
num_features = size(feature_matrix , 2);

if avg==1
    % Task: Averages over the entire feature matrix over all strides
    % needs fixing; not sure if this is a useful option
    features = mean(feature_matrix);
elseif avg == 2
    
    % Task: Averages features per two minutes
    % get size of vector match length of features by ignoring first and
    % last strides
    stride_idx = L_toe_off_idx(2:end-1);
    % calculate number of chunks of 2 minutes 
    % samples per two minutes, sampling rate = 500hz
    n = (2 * 60) / (1/500);
    
    
    % Chunking the time vector and averaging the corresponding stride's
    % features
    
    % Calculate number of chunks (of two minutes)
    num_chunks = floor( length(time)/n )-1 ;

    fprintf('Num Chunks: %i\n', num_chunks); 
    
    % Get indices of each stride start and end
    time_idx = zeros(num_chunks, 2);
    % Index by absolute time
    a = n/2; b = n;
    for i=1:num_chunks
       time_idx(i,:)= [a,b];
       a = a + n; b = b + n;
    end
    
    % Index by stride
    avg_feature_matrix = zeros(num_chunks, num_features);
    avg_metabolics = zeros(num_chunks, 1);
    for chunk=1:num_chunks
        start_idx = time_idx(chunk, 1);
        end_idx = time_idx(chunk, 2);   
        
        vec = (stride_idx >= start_idx & stride_idx <= end_idx);
        index = [find(vec, 1, 'first') , find(vec, 1, 'last')]; 
        
        avg_feature_matrix(chunk,:) = mean(feature_matrix(index(1):index(2), :) ) ;
        avg_metabolics(chunk,:) = mean(metabolics(index(1):index(2), :));
        
        
    end
    features = avg_feature_matrix;
    output = avg_metabolics;
else
    disp('sdfsdfsdfsgdgdsdgsdgsdgsdgs')
end
    
    










