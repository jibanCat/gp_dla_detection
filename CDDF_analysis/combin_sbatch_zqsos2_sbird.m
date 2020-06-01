% combine_sbatch.m: combine sbatch pieces using hardcoded MATLAB for loop

% be sure you get the right normalization range!
set_parameters;

% hard code these init vars since this is a backup script to when sbatch_reunion
% is not working anyway
release           = 'dr12q';
test_set_name     = 'dr12q';

% max_dlas          = 4;
num_quasars_offset =  80000;
num_quasars        =  90000;
num_quasars_split  =     50;
num_files          = floor((num_quasars - num_quasars_offset) / num_quasars_split);
num_final_chunk    =      0;

% the size of sub and sup depends on this
DLA_cut = 20.3;

variables_to_load = {'training_release', 'training_set_name', ...
    'dla_catalog_name', 'release', ...
    'test_set_name', 'test_ind', 'prior_z_qso_increase', ...
    'max_z_cut', 'num_lines', 'all_thing_ids', ...
    'sample_log_posteriors_dla_sub'}; % load this to verify the size

% to fit MATLAB's convention, you start with +1
quasar_start_ind = num_quasars_offset + 1;
quasar_end_ind   = num_quasars_offset + num_quasars_split + 1;

load(sprintf('%s/processed_qsos_zqsos_sbird_%s-%s_%d-%d_norm_%d-%d',             ...
     processed_directory(release), ...
     test_set_name, optTag, ...
     quasar_start_ind, quasar_end_ind, ...
     normalization_min_lambda, normalization_max_lambda), ...
     variables_to_load{:});

% sup and sub have different size than num_dla_samples due to they are acquire by
% logNHI >= 20.3 and logNHI < 20.3; so load these two variables first to verify the size
% load DLA samples from training release
dla_samples_to_load = {'log_nhi_samples'};
load(sprintf('%s/dla_samples', processed_directory(training_release)), ...
    dla_samples_to_load{:});

% check the size is correct
sub20pt3_ind  = (log_nhi_samples < DLA_cut);
sub20pt3_size = sum(sub20pt3_ind);
[~, sub_size] = size(sample_log_posteriors_dla_sub);
assert( sub_size == sub20pt3_size )

% assign the sub and sup size, based on DLA_cut
sub_size = sub20pt3_size;
sup_size = num_dla_samples - sub20pt3_size;

% init large arrays to append
% variables_to_append
min_z_dlas                    = nan(num_quasars, num_dla_samples);
max_z_dlas                    = nan(num_quasars, num_dla_samples);
sample_log_posteriors_no_dla  = nan(num_quasars, num_dla_samples);
sample_log_posteriors_dla     = nan(num_quasars, num_dla_samples);
sample_log_posteriors_dla_sub = nan(num_quasars, sub_size);
sample_log_posteriors_dla_sup = nan(num_quasars, sup_size);
log_posteriors_no_dla         = nan(num_quasars, 1);
log_posteriors_dla            = nan(num_quasars, 1);
log_posteriors_dla_sub        = nan(num_quasars, 1);
log_posteriors_dla_sup        = nan(num_quasars, 1);
z_true                        = nan(num_quasars, 1);
dla_true                      = nan(num_quasars, 1);
z_map                         = nan(num_quasars, 1);
z_dla_map                     = nan(num_quasars, 1);
n_hi_map                      = nan(num_quasars, 1);
log_nhi_map                   = nan(num_quasars, 1);
signal_to_noise               = nan(num_quasars, 1);


% posteriors results which are not in init of process script
model_posteriors = nan(num_quasars, 2);
p_no_dlas        = nan(num_quasars, 1);
p_dlas           = nan(num_quasars, 1);

% variables_to_append : from each small file
variables_to_load = {'min_z_dlas', 'max_z_dlas', ... % you need to save DLA search length to compute CDDF
     'sample_log_posteriors_no_dla', 'sample_log_posteriors_dla', ...
     'sample_log_posteriors_dla_sub', 'sample_log_posteriors_dla_sup', ...
     'log_posteriors_no_dla', 'log_posteriors_dla', ...
     'log_posteriors_dla_sub', 'log_posteriors_dla_sup', ...
     'model_posteriors', 'p_no_dlas', ...
     'p_dlas', 'z_map', 'z_true', 'dla_true', 'z_dla_map', 'n_hi_map', 'log_nhi_map',...
     'signal_to_noise'};
     %'all_posdeferrors', 'all_exceptions', 'qso_ind', % append these variables to complete the catalog

for i = 0:(num_files - 1)
    % only load variables to append to save mamory usage
    % to fit MATLAB's convention, you start with +1
    quasar_start_ind = num_quasars_offset + 1 + num_quasars_split * i;
    quasar_end_ind   = quasar_start_ind + num_quasars_split;

    processed = load(sprintf('%s/processed_qsos_zqsos_sbird_%s-%s_%d-%d_norm_%d-%d',             ...
        processed_directory(release), ...
        test_set_name, optTag, ...
        quasar_start_ind, quasar_end_ind, ...
        normalization_min_lambda, normalization_max_lambda), ...
        variables_to_load{:});
    
    % annoying MATLAB convention: they include both the initial index and the end index when slicing
    qso_start =  i      * num_quasars_split + 1;   % need + 1
    qso_end   = (i + 1) * num_quasars_split;       % no need + 1
    
    % hardcoded to append everything
    % variables_to_append
    min_z_dlas(qso_start:qso_end, :)             = processed.min_z_dlas(:, :);
    max_z_dlas(qso_start:qso_end, :)             = processed.max_z_dlas(:, :);

    sample_log_posteriors_no_dla(qso_start:qso_end, :)  = processed.sample_log_posteriors_no_dla(:, :);
    sample_log_posteriors_dla(qso_start:qso_end, :)     = processed.sample_log_posteriors_dla(:, :);
    sample_log_posteriors_dla_sub(qso_start:qso_end, :) = processed.sample_log_posteriors_dla_sub(:, :);
    sample_log_posteriors_dla_sup(qso_start:qso_end, :) = processed.sample_log_posteriors_dla_sup(:, :);
    
    log_posteriors_no_dla(qso_start:qso_end, 1)  = processed.log_posteriors_no_dla(:, 1);
    log_posteriors_dla(qso_start:qso_end, 1)     = processed.log_posteriors_dla(:, 1);
    log_posteriors_dla_sub(qso_start:qso_end, 1) = processed.log_posteriors_dla_sub(:, 1);
    log_posteriors_dla_sup(qso_start:qso_end, 1) = processed.log_posteriors_dla_sup(:, 1);

    z_true(qso_start:qso_end, 1)          = processed.z_true(:, 1);
    dla_true(qso_start:qso_end, 1)        = processed.dla_true(:, 1);
    z_map(qso_start:qso_end, 1)           = processed.z_map(:, 1);
    z_dla_map(qso_start:qso_end, 1)       = processed.z_dla_map(:, 1);
    n_hi_map(qso_start:qso_end, 1)        = processed.n_hi_map(:, 1);
    log_nhi_map(qso_start:qso_end, 1)     = processed.log_nhi_map(:, 1);
    signal_to_noise(qso_start:qso_end, 1) = processed.signal_to_noise(:, 1);    

    % posteriors results which are not in init of process script
    model_posteriors(qso_start:qso_end, :) = processed.model_posteriors(:, :);
    p_no_dlas(qso_start:qso_end, 1)        = processed.p_no_dlas(:, 1);
    p_dlas(qso_start:qso_end, 1)           = processed.p_dlas(:, 1);

    clear processed    
end

if (num_final_chunk ~= 0)
    % ad-hocly add the final piece of chunck
    quasar_start_ind = num_quasars_offset + num_files * num_quasars_split + 1;
    quasar_end_ind   = quasar_start_ind + num_final_chunk;

    processed = load(sprintf('%s/%s/processed_qsos_zqsos_sbird_%s-%s_%d-%d_norm_%d-%d',             ...
        processed_directory(release), ...
        test_set_name, optTag, ...
        quasar_start_ind, quasar_end_ind, ...
        normalization_min_lambda, normalization_max_lambda), ...
        variables_to_load{:});

    qso_start = num_quasars_offset + num_files * num_quasars_split + 1;               % need + 1
    qso_end   = num_quasars_offset + num_files * num_quasars_split + num_final_chunk; % no need + 1

    % hardcoded to append everything
    % variables_to_append
    min_z_dlas(qso_start:qso_end, :)             = processed.min_z_dlas(:, :);
    max_z_dlas(qso_start:qso_end, :)             = processed.max_z_dlas(:, :);

    sample_log_posteriors_no_dla(qso_start:qso_end, :)  = processed.sample_log_posteriors_no_dla(:, :);
    sample_log_posteriors_dla(qso_start:qso_end, :)     = processed.sample_log_posteriors_dla(:, :);
    sample_log_posteriors_dla_sub(qso_start:qso_end, :) = processed.sample_log_posteriors_dla_sub(:, :);
    sample_log_posteriors_dla_sup(qso_start:qso_end, :) = processed.sample_log_posteriors_dla_sup(:, :);
    
    log_posteriors_no_dla(qso_start:qso_end, 1)  = processed.log_posteriors_no_dla(:, 1);
    log_posteriors_dla(qso_start:qso_end, 1)     = processed.log_posteriors_dla(:, 1);
    log_posteriors_dla_sub(qso_start:qso_end, 1) = processed.log_posteriors_dla_sub(:, 1);
    log_posteriors_dla_sup(qso_start:qso_end, 1) = processed.log_posteriors_dla_sup(:, 1);

    z_true(qso_start:qso_end, 1)          = processed.z_true(:, 1);
    dla_true(qso_start:qso_end, 1)        = processed.dla_true(:, 1);
    z_map(qso_start:qso_end, 1)           = processed.z_map(:, 1);
    z_dla_map(qso_start:qso_end, 1)       = processed.z_dla_map(:, 1);
    n_hi_map(qso_start:qso_end, 1)        = processed.n_hi_map(:, 1);
    log_nhi_map(qso_start:qso_end, 1)     = processed.log_nhi_map(:, 1);
    signal_to_noise(qso_start:qso_end, 1) = processed.signal_to_noise(:, 1);    

    % posteriors results which are not in init of process script
    model_posteriors(qso_start:qso_end, :) = processed.model_posteriors(:, :);
    p_no_dlas(qso_start:qso_end, 1)        = processed.p_no_dlas(:, 1);
    p_dlas(qso_start:qso_end, 1)           = processed.p_dlas(:, 1);

    clear processed    
end

fprintf_debug('Number of isnan : %d; number of all_exceptions : %d', ...
    sum(isnan(log_posteriors_no_dla)), sum(~isnan(log_posteriors_dla_sup)));

% save results
% exclude base_sample_inds and sample_log_likelihoods_dla
variables_to_save = {'training_release', 'training_set_name', ...
                     'dla_catalog_name', 'release', ...
                     'test_set_name', 'test_ind', 'prior_z_qso_increase', ...
                     'max_z_cut', 'num_lines', 'min_z_dlas', 'max_z_dlas', ... % you need to save DLA search length to compute CDDF
                     'sample_log_posteriors_no_dla', 'sample_log_posteriors_dla', ...
                     'sample_log_posteriors_dla_sub', 'sample_log_posteriors_dla_sup', ...
                     'log_posteriors_no_dla', 'log_posteriors_dla', ...
                     'log_posteriors_dla_sub', 'log_posteriors_dla_sup', ...
                     'model_posteriors', 'p_no_dlas', ...
                     'p_dlas', 'z_map', 'z_true', 'dla_true', 'z_dla_map', 'n_hi_map', 'log_nhi_map',...
                     'signal_to_noise', 'all_thing_ids'};

filename = sprintf('%s/processed_qsos_zqsos_sbird_%s-%s_norm_%d-%d', ...
                    processed_directory(release), ...
                    test_set_name, optTag, ...
                    normalization_min_lambda, normalization_max_lambda);

save(filename, variables_to_save{:}, '-v7.3');
