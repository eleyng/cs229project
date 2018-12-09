close all; clear all; clc;


% Testing the whole pipeline just on TBXR 11/14
%{
data_dir_11_07 = '~/229_project_data/TBXR/11_07';         % feed this in as input 
data_dir_11_14 = '~/229_project_data/TBXR/11_14';         % feed this in as input 
data_dir_11_19 = '~/229_project_data/TBXR/11_19';         % feed this in as input 
data_dir_11_28 = '~/229_project_data/TBXR/11_28';         % feed this in as input 

[data_11_07, metabolics_11_07] = process_day(data_dir_11_07); pause(1); drawnow; 
[data_11_14, metabolics_11_14] = process_day(data_dir_11_14);  pause(1); drawnow; 
[data_11_19, metabolics_11_19] = process_day(data_dir_11_19);  pause(1); drawnow; 
[data_11_28, metabolics_11_28] = process_day(data_dir_11_28);  pause(1); drawnow; 
data = [data_11_07; data_11_14; data_11_19; data_11_28]; 
full_metabolics = [metabolics_11_07; metabolics_11_14; metabolics_11_19; metabolics_11_28]; 


%% create row vector with the median from everything 
%        step_data = [Fz_med, peak_df_med, peak_pf_med, step_time_med]; 

% Looking at Left and Right ground reaction 
figure, 
plot(data_11_07(:, 1), metabolics_11_07, 'mx'), hold on 
plot(data_11_07(:, 5), metabolics_11_07, 'mo')
plot(data_11_14(:, 1), metabolics_11_14, 'bx')
plot(data_11_14(:, 5), metabolics_11_14, 'bo')    
plot(data_11_19(:, 1), metabolics_11_19, 'rx')
plot(data_11_19(:, 5), metabolics_11_19, 'ro')
plot(data_11_28(:, 1), metabolics_11_28, 'gx') 
plot(data_11_28(:, 5), metabolics_11_28, 'go')
title('peak groudn reaction')
hold off 

figure, 
plot(data_11_07(:, 2), metabolics_11_07, 'mx'), hold on 
plot(data_11_07(:, 6), metabolics_11_07, 'mo')
plot(data_11_14(:, 2), metabolics_11_14, 'bx')
plot(data_11_14(:, 6), metabolics_11_14, 'bo')    
plot(data_11_19(:, 2), metabolics_11_19, 'rx')
plot(data_11_19(:, 6), metabolics_11_19, 'ro')
plot(data_11_28(:, 2), metabolics_11_28, 'gx') 
plot(data_11_28(:, 6), metabolics_11_28, 'go')
title('peak df ')
hold off 
figure, 
plot(data_11_07(:, 3), metabolics_11_07, 'mx'), hold on 
plot(data_11_07(:, 7), metabolics_11_07, 'mo')
plot(data_11_14(:, 3), metabolics_11_14, 'bx')
plot(data_11_14(:, 7), metabolics_11_14, 'bo')    
plot(data_11_19(:, 3), metabolics_11_19, 'rx')
plot(data_11_19(:, 7), metabolics_11_19, 'ro')
plot(data_11_28(:, 3), metabolics_11_28, 'gx') 
plot(data_11_28(:, 7), metabolics_11_28, 'go')
title('peak pf ')
hold off 


figure, 
plot(data_11_07(:, 4), metabolics_11_07, 'mx'), hold on 
plot(data_11_07(:, 8), metabolics_11_07, 'mo')
plot(data_11_14(:, 4), metabolics_11_14, 'bx')
plot(data_11_14(:, 8), metabolics_11_14, 'bo')    
plot(data_11_19(:, 4), metabolics_11_19, 'rx')
plot(data_11_19(:, 8), metabolics_11_19, 'ro')
plot(data_11_28(:, 4), metabolics_11_28, 'gx') 
plot(data_11_28(:, 8), metabolics_11_28, 'go')
title('step time ')
hold off 


%}


% Eley Data 

data_dir_10_07 = '~/229_project_data/PAWN/10_07';         % feed this in as input 
data_dir_10_09 = '~/229_project_data/PAWN/10_09';         % feed this in as input 
data_dir_10_19 = '~/229_project_data/PAWN/10_19';         % feed this in as input 
data_dir_10_25 = '~/229_project_data/PAWN/10_25';         % feed this in as input 
data_dir_11_04 = '~/229_project_data/PAWN/11_04';         % feed this in as input 

[data_10_07, metabolics_10_07] = process_day(data_dir_10_07); pause(1); drawnow; 
[data_10_09, metabolics_10_09] = process_day(data_dir_10_09); pause(1); drawnow; 
[data_10_19, metabolics_10_19] = process_day(data_dir_10_19);  pause(1); drawnow; 
[data_10_25, metabolics_10_25] = process_day(data_dir_10_25);  pause(1); drawnow; 
[data_11_04, metabolics_11_04] = process_day(data_dir_11_04);  pause(1); drawnow; 

eley_data = [data_10_07; data_10_09; data_10_19; data_10_25; data_11_04]; 
eley_metabolics = [metabolics_10_07; metabolics_10_09; metabolics_10_19; metabolics_10_25; metabolics_11_04]; 


% Michael Data 
data_dir_10_26 = '~/229_project_data/PBPK/10_26';         % feed this in as input 
data_dir_10_28 = '~/229_project_data/PBPK/10_28';         % feed this in as input 
data_dir_11_02 = '~/229_project_data/PBPK/11_02';         % feed this in as input 
data_dir_11_04 = '~/229_project_data/PBPK/11_04';         % feed this in as input 
data_dir_11_08 = '~/229_project_data/PBPK/11_08';         % feed this in as input 

[data_10_26, metabolics_10_26] = process_day(data_dir_10_26); pause(1); drawnow; 
[data_10_28, metabolics_10_28] = process_day(data_dir_10_28); pause(1); drawnow; 
[data_11_02, metabolics_11_02] = process_day(data_dir_11_02);  pause(1); drawnow; 
[data_11_04, metabolics_11_04] = process_day(data_dir_11_04);  pause(1); drawnow; 
[data_11_08, metabolics_11_08] = process_day(data_dir_11_08);  pause(1); drawnow; 

michael_data = [data_10_26; data_10_28; data_11_02; data_11_04; data_11_08]; 
michael_metabolics = [metabolics_10_26; metabolics_10_28; metabolics_11_02; metabolics_11_04; metabolics_11_08]; 

%data(chunk, :) = [left_step_data, right_step_data, med_step_width, rms_emg, control_params]; 
% step_data = [Fz_med, peak_df_med, peak_pf_med, step_time_med]; 
data_labels = {'LFz_peak', 'L_peak_df', 'L_peak_pf', 'L_step_time',...
               'RFz_peak', 'R_peak_df', 'R_peak_pf', 'R_step_time', 'step_width',...
               'emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8',...
               'emg9', 'emg10', 'emg11', 'emg12', 'emg13', 'emg14', 'emg15', 'emg16',...
               'ctrl1', 'ctrl2', 'ctrl3', 'ctrl4'}; 


close all; 
save('processed_data.mat', 'eley_data', 'eley_metabolics', 'michael_data', 'michael_metabolics', 'data_labels'); 

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

% pca all 
num_pcs_all = 8; 
all_shift = bsxfun(@minus, data, mean(data));
all_unit_var = bsxfun(@rdivide, all_shift, sqrt(var(all_shift)));
[coeff_all, score_all, latent_all, tsquared_all, explained_all, mu_all] = pca(all_unit_var);
U_all = coeff_all(:, 1:num_pcs_all);
pca_all = all_unit_var * U_all; 

% find mean RMS for constant prediction 

constant_pred_rms = rms(full_metabolics - mean(full_metabolics));
constant_pred_mse = constant_pred_rms^2; 
display(constant_pred_rms); 
display(constant_pred_mse); 

%}
function [data, normalized_metabolics] = process_day(data_dir)


    fprintf('Processing directory: %s\n', data_dir); 

    dt = 1/500; % 500 Hz sampling 

    %% Quiet Standing
    %
    %   Want
    %       - baseline metabolics
    %       - ground reaction force 


    load(fullfile(data_dir,'mat', 'qs.mat')); 

    baseline_met = mean(met); 
    weight = median(RFz + LFz);      % for normalizing 

    display(weight)

    %% Normal Walking 1 - TBXRDE
    %       - get metabolics 
    %       - get PEAK emgs signals  

    num_emg = 16; 
    load(fullfile(data_dir,'mat', 'nw1.mat')); 
    nw1_full_met = met; 
    nw1_end_met = nw1_full_met(round(length(nw1_full_met)/2):end); 
    nw_met1 = mean(nw1_end_met);  % normal walking metabolic cost 

    %% Normal Walking 2 - TBXRDF 
    load(fullfile(data_dir,'mat', 'nw2.mat')); 
    nw2_full_met = met; 
    nw2_end_met = nw2_full_met(round(length(nw2_full_met)/2):end); 
    nw_met2 = mean(nw2_end_met);  % normal walking metabolic cost 

    nw_met = (nw_met1 + nw_met2)/2; 

    

    nw_start_row = 150000;
    vicon_nw1 = csvread(fullfile(data_dir,'EMG', 'nw1.csv'), nw_start_row, 1, [nw_start_row, 1, 360000, 47]);
    vicon_nw2 = csvread(fullfile(data_dir,'EMG', 'nw2.csv'), nw_start_row, 1, [nw_start_row, 1, 360000, 47]);




    emg_nw1 = vicon_nw1(:, (end-(num_emg-1)):end); 
    emg_nw2 = vicon_nw2(:, (end-(num_emg-1)):end); 

    emg_nw1_filt = emg_filter(emg_nw1);
    emg_nw2_filt = emg_filter(emg_nw2);

    emg_filt = [emg_nw1_filt; emg_nw2_filt]; 

    emg_peaks = max(emg_filt, [], 1); 




    % Now loop through opt data 

    % absolute shitshow of data so need to be smart here 
    parts = strsplit(data_dir, '/'); 
    parent_dir = parts{end -1}; 

    csv_list = {};
    if strcmp(parent_dir, 'TBXR')
        load(fullfile(data_dir,'mat', 'opti.mat')); 
        evalvars; % MAKE BETTER PIPELINE 

        
        for i = 2:37
            csv_list{end + 1} = sprintf('opt%d.csv', i);
        end 

    elseif strcmp(parent_dir, 'PAWN')

        opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
        if (strcmp(parts{end}, '11_04'))
            opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
            load(opti1); 
        else 
            opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
            opti2 = fullfile(data_dir,'mat', 'opt2.mat'); 
            combine_opti(opti1, opti2);

            % stupid processing, plz kill me 
            twos = find(condNo == 2);
            cn_idx = 1; 
            while (cn_idx < length(condNo))

                % if cond no not 2 - skip till when it is 
                while (cn_idx < length(condNo)) && (condNo(cn_idx) ~= 2)
                    cn_idx = cn_idx + 1; 
                end 

                % so it equals two or we finished 

                if (cn_idx < length(condNo)) && (condNo(cn_idx) == 2)  
                    % find how long until it doesnt equal 2
                    next_idx = cn_idx;
                    while (condNo(next_idx) == 2)  && (next_idx < length(condNo))
                        next_idx = next_idx + 1; 
                    end      

                    two_end_idx = next_idx - 1; 

                    if (two_end_idx - cn_idx) < (1.5 * 60/dt)   % conditions should be 2 minutes 
                        condNo(cn_idx:two_end_idx) = 1; 
                    end 
                    cn_idx = two_end_idx; 
                end 
                cn_idx = cn_idx + 1; 
            end 
        end 

        if (strcmp(parts{end}, '10_07'))
            for i = 3:38
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        elseif (strcmp(parts{end}, '10_09'))
            for i = 2:37       % TODO -- double check because there is opt 0? 
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        elseif (strcmp(parts{end}, '10_19'))
            for i = 1:36        % TODO unclear whatsup here 
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        elseif (strcmp(parts{end}, '10_25'))
            for i = 2:37
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        elseif (strcmp(parts{end}, '11_04'))
            for i = 2:37
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        end 


    elseif strcmp(parent_dir, 'PBPK')

        opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
        if (strcmp(parts{end}, '10_26')) ||  (strcmp(parts{end}, '10_28')) ||  (strcmp(parts{end}, '11_02'))  ||  (strcmp(parts{end}, '11_04'))
            opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
            load(opti1); 
        else    % 11_08
            opti1 = fullfile(data_dir,'mat', 'opt1.mat'); 
            opti2 = fullfile(data_dir,'mat', 'opt2.mat'); 
            combine_opti(opti1, opti2);

            % stupid processing, plz kill me 
            twos = find(condNo == 2);
            cn_idx = 1; 
            while (cn_idx < length(condNo))

                % if cond no not 2 - skip till when it is 
                while (cn_idx < length(condNo)) && (condNo(cn_idx) ~= 2)
                    cn_idx = cn_idx + 1; 
                end 

                % so it equals two or we finished 

                if (cn_idx < length(condNo)) && (condNo(cn_idx) == 2)  
                    % find how long until it doesnt equal 2
                    next_idx = cn_idx;
                    while (condNo(next_idx) == 2)  && (next_idx < length(condNo))
                        next_idx = next_idx + 1; 
                    end      

                    two_end_idx = next_idx - 1; 

                    if (two_end_idx - cn_idx) < (1.5 * 60/dt)   % conditions should be 2 minutes 
                        condNo(cn_idx:two_end_idx) = 1; 
                    end 
                    cn_idx = two_end_idx; 
                end 
                cn_idx = cn_idx + 1; 
            end 
        end 

        if (strcmp(parts{end}, '10_26')) ||  (strcmp(parts{end}, '10_28'))  ||  (strcmp(parts{end}, '11_02'))  ||  (strcmp(parts{end}, '11_04'))
            for i = 2:37
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        elseif (strcmp(parts{end}, '11_08'))
            for i = [2:6, 10:40]       % TODO -- double check because there is opt 0? 
                csv_list{end + 1} = sprintf('opt%d.csv', i);
            end 
        end 





    end 

    
    time = time - time(1); 

    

    % Center of pressure 
    v_tm = 1.25;    % treadmill speed, m/s 

    h = 15e-3; % heigh of treadmill plane above sensor belt
    L_xp = (-h * LFx - LMy)./LFz; 
    L_yp = (h * LFy + LMx)./LFz; 

    R_xp = (-h * RFx - RMy)./RFz; 
    R_yp = (h * RFy + RMx)./RFz; 



    unshifted_cost = cost; 


    %% shift the metabolcis data so that it actually lines up 
    shift_amount = round(2 * 60/dt);    % 2 minutes shift 
    cost_end = cost(end); 
    cost = circshift(cost, -shift_amount); 
    cost((end - shift_amount + 1):end) = cost_end; 

    % normalize 
    LFz = LFz/weight;
    RFz = RFz/weight; 
    nw_met = nw_met - baseline_met; 

    %normalized_met = (met - baseline_met)/nw_met; 

    cost = cost - baseline_met; 
    normalized_cost = cost/nw_met;      % normalize by normal walking condition 


    % need to find end end Idx based of condNo variable because of split data 
    pts_per_chunk = round(30/dt); 
    offset = find(diff(normalized_cost) ~= 0, 1);
    chunk_size = round(2 * 60/dt);  % kill first two minutes 
    num_chunks = 36;
    
    max_idx = length(time);

   

    normalized_metabolics = zeros(num_chunks, 1); 

    features_per_leg = 4; % for now 
    num_control = 4; 
    % + 1 for step width
    num_features = 2 * features_per_leg + 1 + num_emg + num_control;   % [Fz_med, peak_df_med, peak_pf_med, step_time_med]; 
    data = zeros(num_chunks, num_features);     

    % sanity check
    figure, plot(time, normalized_cost, 'k'), hold on
    plot(time, unshifted_cost/max(unshifted_cost), 'b')
    plot(time(offset), normalized_cost(offset), 'rx')
    title(data_dir); 
    hold off



    

    display(offset); 


    condNoIdx = find(condNo == 2, 1);
    display(condNoIdx)
    
    end_idxs = find(diff(condNo) ~= 0);
    if (end_idxs(1) < chunk_size)
        end_idxs = end_idxs(2:end); 
    end  




    figure, plot(time, condNo); hold on 
    plot(time(end_idxs), condNo(end_idxs), 'rx')
    hold off; drawnow;




    for chunk = 1:num_chunks 


        % 30 second interval 
        %end_idx = min(offset + (chunk * chunk_size), max_idx); 
        end_idx = end_idxs(chunk); 
        start_idx = end_idx - pts_per_chunk;

        % load in the vicon data -- process the emg 
        opt_file = csv_list{chunk}; 

        vicon_file = fullfile(data_dir,'EMG', opt_file); 
        display(vicon_file); 
        

        if (chunk == num_chunks)
            % for the last, only prcoess the end of it 
            %vicon_chunk = csvread(vicon_file, 6, 1);    % if get issue -- will hard code 

            try
                vicon_chunk = csvread(vicon_file, 6, 1, [6, 1, 60004, 47]);
                vicon_rows = size(vicon_chunk, 1); 
                vicon_chunk(1:ceil(vicon_rows/2), :);
            catch
                vicon_chunk = csvread(vicon_file, 6, 1);    % because 11_02 data short for michael 
                vicon_rows = size(vicon_chunk, 1); 
                vicon_chunk(1:ceil(vicon_rows/2), :);
            end 

        else
            vicon_chunk = csvread(vicon_file, 6, 1, [6, 1, 30005, 47]);
            vicon_rows = size(vicon_chunk, 1); 
        end 

        % get control params 
        cmap1_med = median(cmap1(start_idx:end_idx));
        cmap2_med = median(cmap2(start_idx:end_idx));
        cmap3_med = median(cmap3(start_idx:end_idx));
        cmap4_med = median(cmap4(start_idx:end_idx));

        control_params = [cmap1_med, cmap2_med, cmap3_med, cmap4_med]; 

        raw_emg = vicon_chunk(:, (end-(num_emg-1)):end); 
        filtered_emg = emg_filter(raw_emg); 
        normalized_emg = bsxfun(@rdivide, filtered_emg, emg_peaks);
        rms_emg = rms(normalized_emg); 



        %% ----------- Outputs --------------------------

        % left step info 
        [peak_force_idxsL, left_step_data] = process_steps(start_idx, end_idx, phaseL, LFz, posaL);
        [peak_force_idxsR, right_step_data] = process_steps(start_idx, end_idx, phaseR, RFz, posaR); 

        if (any(isnan(left_step_data)) || any(isnan(right_step_data)))
            data(chunk, :) = nan; 

        else

            if (length(peak_force_idxsL) > length(peak_force_idxsR))
                peak_force_idxsL = peak_force_idxsL(1:length(peak_force_idxsR));
            elseif (length(peak_force_idxsL) < length(peak_force_idxsR))
                peak_force_idxsR = peak_force_idxsR(1:length(peak_force_idxsL));
            end 



            % now same length -- start with whichever has smaller index 
            step_widths = length(peak_force_idxsL); 
            
            %{
            if (peak_force_idxsL(1) < peak_force_idxsR(1))
                for i = 1:length(peak_force_idxsL)
                    step_widths(i) = 
                end 
            else
                for i = 1:length(peak_force_idxsL)
                    step_widths(i) =  
                end
            end 
            %}

            for i = 1:length(peak_force_idxsL)
               step_widths(i) = L_xp(peak_force_idxsL(i)) -  R_xp(peak_force_idxsR(i)); 
            end

            %tmp = peak_force_idxsL - peak_force_idxsR;
            %display(tmp)
            %display(peak_force_idxsL)
            %display(peak_force_idxsR)
            %display(step_widths)
            % want step width 
            % define as width between left rising edge 
            med_step_width = median(step_widths); 
            %display(med_step_width)

            normalized_metabolics(chunk) = mean(normalized_cost(start_idx:end_idx)); 
            data(chunk, :) = [left_step_data, right_step_data, med_step_width, rms_emg, control_params]; 
        end     


    end 

    figure, plot(normalized_metabolics)
    title('full norm met for day')
    good_idxs = all(~isnan(data),2); 
    data = data(good_idxs,:); 
    normalized_metabolics = normalized_metabolics(good_idxs,:); 

end     % end process_day

function combine_opti(opti1, opti2)
        vars1 = load(opti1);
        vars2 = load(opti2); 
        names = fieldnames(vars1); 
        dt = 1/500; 
        for i = 1:length(names)
            
            vars1.time = vars1.time - vars1.time(1); 
            vars2.time = vars2.time - vars2.time(1); 
            vars2.time = vars2.time + dt + vars1.time(end); 
            tmp1 = getfield(vars1, names{i}); 
            tmp2 = getfield(vars2, names{i});
            tmp = [tmp1; tmp2];

            disp(names{i})
            disp(length(tmp1))
            disp(length(tmp2))
            disp(length(tmp))
            
            disp(names{i})
            %disp(tmp)

        assignin('caller', names{i}, tmp); 
    end 
end 
% opt 37 is special condition 

% process step data in a chunk 

function [Fz_peak_idxs, step_data] = process_steps(start_idx, end_idx, phase, Fz, posa)

    % get number of steps from the heel switch so we can allocate arrays 
    % define step as heel switch rising edge to rising edge 
    % number of steps is number of rising edges - 1 
    dt = 1/500; % 500 Hz sampling 


    display(start_idx)
    display(end_idx)

    step_idxs = (start_idx - 1) + find(diff(phase(start_idx:end_idx)) == 1) ; 
    

    num_steps = length(step_idxs) - 1;
    %display(num_steps)

    Fz_peak = zeros(num_steps, 1); 
    Fz_peak_idxs = zeros(num_steps, 1);     % for debug 
    step_time = zeros(num_steps, 1);
    peak_pf = zeros(num_steps, 1);          % peak plantarlfexion angle 
    peak_df = zeros(num_steps, 1); 

    for i = 1:num_steps

        % find where toe comes off 
        step_start_idx = step_idxs(i);
        step_end_idx = step_idxs(i + 1) - 1; 
        toe_off_idx = step_start_idx + find(diff(phase(step_start_idx:step_end_idx)) == -1, 1) - 1;
        
        %display(heel_off_idx)

        peak_check_idx = step_start_idx + round((toe_off_idx - step_start_idx)/2); 

        % Peak Force
        [Fz_peak(i), tmp] = max(Fz(peak_check_idx:step_end_idx)); 
        Fz_peak_idxs(i) = tmp + peak_check_idx;

        % Peak plantarflexion
        peak_pf(i) =  max(posa(step_start_idx:step_end_idx)); % min for dorsi

         % Peak dorsiflexion
        peak_df(i) =  min(posa(step_start_idx:step_end_idx)); % min for dorsi


        step_time(i) = dt * (step_end_idx - step_start_idx);
    end     




    % debug code 
    
    
    
    if isempty(step_idxs) || (median(step_time) < 0.6) || (num_steps < 15) 

        figure, 
        t = 1:length(Fz); 
        plot(t(start_idx:end_idx), Fz(start_idx:end_idx), 'k'); hold on 
        plot(t(start_idx:end_idx), phase(start_idx:end_idx), 'g');
        plot(t(Fz_peak_idxs), Fz(Fz_peak_idxs), 'ro'); 
        hold off 
        step_data = nan;    % flag for nonsense
    else

        %% calculate median for feautures 
        Fz_med = median(Fz_peak);
        peak_pf_med = median(peak_pf); 
        peak_df_med = median(peak_df); 
        step_time_med = median(step_time);

        %display(step_time_med)

        %% create row vector with the median from everything 
        step_data = [Fz_med, peak_df_med, peak_pf_med, step_time_med]; 
    end     
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