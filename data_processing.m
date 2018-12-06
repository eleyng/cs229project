close all; clear all; clc;


% Testing the whole pipeline just on TBXR 11/14

data_dir_11_14 = '~/229_project_data/TBXR/11_14';         % feed this in as input 
data_dir_11_19 = '~/229_project_data/TBXR/11_19';         % feed this in as input 
data_dir_11_28 = '~/229_project_data/TBXR/11_28';         % feed this in as input 

[data_11_14, metabolics_11_14] = process_day(data_dir_11_14); 
[data_11_19, metabolics_11_19] = process_day(data_dir_11_19); 
[data_11_28, metabolics_11_28] = process_day(data_dir_11_28); 

data = [data_11_14; data_11_19; data_11_28]; 
full_metabolics = [metabolics_11_14; metabolics_11_19; metabolics_11_28]; 



fucked_up_idxs = find(full_metabolics < 0.25, 3); 
data(fucked_up_idxs, :) = []; 
full_metabolics(fucked_up_idxs) = []; 


data_no_emg = data(:, 1:(end-16)); 
emg_only = data(:, (end-15):end);   % for PCA, woot woot!

left_emg_idx = [1:4, 9:12]; 
right_emg_idx = [5:8, 13:16]; 













%% pca on the emg becase "machine learning"

% take all the emg make zero mean and unit variance
emg_for_pca = bsxfun(@minus, emg_only, mean(emg_only));
emg_for_pca = bsxfun(@rdivide, emg_for_pca, sqrt(var(emg_for_pca)))
[coeff,score,latent, tsquared, explained, mu] = pca(emg_for_pca)

% 
num_pcs = 6; 

U = coeff(:, 1:num_pcs);
pca_emg = emg_for_pca * U; 





function [data, normalized_metabolics] = process_day(data_dir)


    %% Quiet Standing
    %
    %   Want
    %       - baseline metabolics
    %       - ground reaction force 


    load(fullfile(data_dir,'mat', 'qs.mat')); 

    baseline_met = mean(met); 
    weight = mean(RFz + LFz);      % for normalizing 



    %% Normal Walking 1 - TBXRDE
    %       - get metabolics 
    %       - get PEAK emgs signals  

    num_emg = 16; 
    load(fullfile(data_dir,'mat', 'nw1.mat')); 
    nw_met1 = mean(cost);  % normal walking metabolic cost 

    %% Normal Walking 2 - TBXRDF 
    load(fullfile(data_dir,'mat', 'nw2.mat')); 
    nw_met2 = mean(cost);  % normal walking metabolic cost 
    nw_met = (nw_met1 + nw_met2)/2;         % looks like they are the same though anyway

    vicon_nw1 = csvread(fullfile(data_dir,'EMG', 'nw1.csv'), 6, 1);
    vicon_nw2 = csvread(fullfile(data_dir,'EMG', 'nw2.csv'), 6, 1);
    emg_nw1 = vicon_nw1(:, (end-(num_emg-1)):end); 
    emg_nw2 = vicon_nw2(:, (end-(num_emg-1)):end); 

    emg_nw1_filt = emg_filter(emg_nw1);
    emg_nw2_filt = emg_filter(emg_nw2);

    emg_filt = [emg_nw1_filt; emg_nw2_filt]; 

    emg_peaks = max(emg_filt, [], 1); 








    % Now loop through opt data 
    load(fullfile(data_dir,'mat', 'opti.mat')); 
    evalvars; % MAKE BETTER PIPELINE 


    % normalize 
    LFz = LFz/weight;
    RFz = RFz/weight; 
    nw_met = nw_met - baseline_met; 

    normalized_met = (met - baseline_met)/nw_met; 

    cost = cost - baseline_met; 
    normalized_cost = cost/nw_met;      % normalize by normal walking condition 


    dt = 1/500; % 500 Hz sampling 

    % 


    % kill first 2 minutes -- start with opt 2 




    % clean up mechanical noise on heel switch 
    %{
    filt_size = 10; 
    idx = filt_size/2; 
    while (idx + filt_size/2) < length(hsL)
        hsL(((idx - filt_size/2) + 1):(idx + filt_size/2)) = mode(hsL(((idx - filt_size/2) + 1):(idx + filt_size/2)));
        hsR(((idx - filt_size/2) + 1):(idx + filt_size/2)) = mode(hsR(((idx - filt_size/2) + 1):(idx + filt_size/2)));
        idx = idx + filt_size - 1; 
    end 
    %}


    [B_low,A_low] = butter(4,.01,'low');
    hsL_filt = filtfilt(B_low, A_low, hsL);
    hsR_filt = filtfilt(B_low, A_low, hsR);
    hsL = round(hsL_filt); 
    hsR = round(hsR_filt); 


    num_chunks = 36;
    t0 = time(1);
    max_idx = length(time);

    pts_per_chunk = round(30/dt); 
    offset = round(2 * 60/dt);  % kill first two minutes 


    normalized_metabolics = zeros(num_chunks, 1); 

    features_per_leg = 3; % for now 
    num_features = 2 * features_per_leg + num_emg; 
    data = zeros(num_chunks, num_features); 

    for chunk = 1:num_chunks % num_chunks        % TODO == put back  






        % 30 second interval 
        end_idx = min(offset + (chunk * offset), max_idx); 
        start_idx = end_idx - pts_per_chunk;

        % load in the vicon data -- process the emg 
        opt_file = sprintf('opt%d.csv', chunk+1);
        vicon_chunk = csvread(fullfile(data_dir,'EMG', opt_file), 6, 1);
        vicon_rows = size(vicon_chunk, 1); 

        if (chunk == num_chunks)
            % for the last, only prcoess the end of it 
            vicon_chunk((floor(vicon_rows/2)):end, :);
        else
            vicon_chunk(1:ceil(vicon_rows/2), :);
        end 

        raw_emg = vicon_chunk(:, (end-(num_emg-1)):end); 
        filtered_emg = emg_filter(raw_emg); 
        normalized_emg = bsxfun(@rdivide, filtered_emg, emg_peaks);
        rms_emg = rms(normalized_emg); 



        %% ----------- Outputs --------------------------

        left_step_data = process_steps(start_idx, end_idx, hsL, LFz, posaL);
        right_step_data = process_steps(start_idx, end_idx, hsR, RFz, posaR); 

        normalized_metabolics(chunk) = mean(normalized_cost(start_idx:end_idx)); 

        if (normalized_metabolics(chunk) == 0)
            normalized_metabolics(chunk) = median(normalized_met(start_idx:end_idx)); 
        end
        data(chunk, :) = [left_step_data, right_step_data, rms_emg]; 


    end 

end     % end process_day



% opt 37 is special condition 

% process step data in a chunk 

function [step_data] = process_steps(start_idx, end_idx, hs, Fz, posa)

    % get number of steps from the heel switch so we can allocate arrays 
    % define step as heel switch rising edge to rising edge 
    % number of steps is number of rising edges - 1 
    dt = 1/500; % 500 Hz sampling 
    step_idxs = (start_idx - 1) + find(diff(hs(start_idx:end_idx)) == 1) ; 
    

    num_steps = length(step_idxs) - 1;
    %display(num_steps)

    Fz_peak = zeros(num_steps, 1); 
    Fz_peak_idxs = zeros(num_steps, 1);     % for debug 
    step_time = zeros(num_steps, 1);
    peak_pf = zeros(num_steps, 1);          % peak plantarlfexion angle 


    for i = 1:num_steps

        % find where heel comes off 
        step_start_idx = step_idxs(i);
        step_end_idx = step_idxs(i + 1) - 1; 




        heel_off_idx = step_start_idx + find(diff(hs(step_start_idx:step_end_idx)) == -1, 1) - 1;
        %display(heel_off_idx)

        % Peak Force
        [Fz_peak(i), tmp] = max(Fz(heel_off_idx:step_end_idx)); 
        Fz_peak_idxs(i) = tmp + heel_off_idx;

        % Peak plantarflexion
        peak_pf(i) =  max(posa(step_start_idx:step_end_idx)); % min for dorsi


        step_time(i) = dt * (step_end_idx - step_start_idx);
    end     


    % debug code 
    t = 1:length(Fz); 
    %figure, 
    %plot(t(start_idx:end_idx), Fz(start_idx:end_idx), 'k'); hold on 
    %plot(t(start_idx:end_idx), hs(start_idx:end_idx), 'b');
    %plot(t(Fz_peak_idxs), Fz(Fz_peak_idxs), 'ro'); 
    %hold off 




    %% calculate median for feautures 
    Fz_med = median(Fz_peak);
    peak_pf_med = median(peak_pf); 
    step_time_med = median(step_time);


    %% create row vector with the median from everything 
    step_data = [Fz_med, peak_pf_med, step_time_med]; 
end     






% Filters EMG data, assumes data is stored in columns 

function [filtered] = emg_filter(raw_emg)
    [B_high,A_high] = butter(4,.02,'high');
    [B_low,A_low] = butter(4,.01,'low');

    filtered = zeros(size(raw_emg)); 
    num_signal = size(raw_emg, 2);
    for i = 1:num_signal
        signal = raw_emg(:, i);
        signal = filtfilt(B_high, A_high, signal); 
        signal = abs(signal); 
        filtered(:, i) = filtfilt(B_low, A_low, signal);
    end 
end 