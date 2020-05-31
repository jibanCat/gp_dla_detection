% combine_sbatch.m: combine sbatch pieces using hardcoded MATLAB for loop

% hard code these init vars since this is a backup script to when sbatch_reunion
% is not working anyway
release           = 'dr14q';
test_set_name     = 'dr14q';

max_dlas          = 4;
num_quasars       = 188355;
num_quasars_split = 1000;
num_files         = 188;
num_final_chunk   = 355;


variables_to_load = {'training_release', 'training_set_name', ...
                     'dla_catalog_name', 'prior_ind', 'release', ...
                     'test_set_name', 'prior_z_qso_increase', 'k', ...
                     'normalization_min_lambda', 'normalization_max_lambda', ...
                     'min_z_cut', 'max_z_cut', 'num_dla_samples', ...
                     'num_lines', 'test_ind'};

load(sprintf('%s/processed_qsos_multi_lls_occam_lyseries_variance_kim_lyb_a03_dr14q_0-1000',             ...
     processed_directory(release)), ...
     variables_to_load{:});

% init large arrays to append
% variables_to_append
min_z_dlas                 = nan(num_quasars, 1);
max_z_dlas                 = nan(num_quasars, 1);
log_priors_no_dla          = nan(num_quasars, 1);
log_priors_dla             = nan(num_quasars, max_dlas);
log_likelihoods_no_dla     = nan(num_quasars, 1);
sample_log_likelihoods_dla = nan(num_quasars, num_dla_samples, max_dlas);
base_sample_inds           = zeros(num_quasars, num_dla_samples, max_dlas - 1, 'uint32');
log_likelihoods_dla        = nan(num_quasars, max_dlas);
log_posteriors_no_dla      = nan(num_quasars, 1);
log_posteriors_dla         = nan(num_quasars, max_dlas);

% initialize lls results
log_likelihoods_lls        = nan(num_quasars, 1);
log_posteriors_lls         = nan(num_quasars, 1);
log_priors_lls             = nan(num_quasars, 1);
sample_log_likelihoods_lls = nan(num_quasars, num_dla_samples);

% save maps: add the initilizations of MAP values
% N * (1~k models) * (1~k MAP dlas)
MAP_z_dlas   = nan(num_quasars, max_dlas, max_dlas);
MAP_log_nhis = nan(num_quasars, max_dlas, max_dlas);

all_exceptions = nan(num_quasars, 1);

% posteriors results which are not in init of process script
model_posteriors = nan(num_quasars, max_dlas + 2);
p_no_dlas        = nan(num_quasars, 1);
p_dlas           = nan(num_quasars, 1);
p_lls            = nan(num_quasars, 1);

% variables_to_append : from each small file
variables_to_append = {'min_z_dlas', 'max_z_dlas', ...
    'sample_log_likelihoods_dla', 'base_sample_inds', ...
    'log_priors_no_dla', 'log_priors_dla', 'log_priors_lls', ...
    'log_likelihoods_no_dla', 'MAP_z_dlas', 'MAP_log_nhis', ...
    'log_likelihoods_dla', 'log_likelihoods_lls', ...
    'log_posteriors_no_dla', 'log_posteriors_dla', 'log_posteriors_lls', ...
    'model_posteriors', 'p_no_dlas', 'p_dlas', 'p_lls', ...
    'all_exceptions', 'sample_log_likelihoods_lls'};


for i = 0:(num_files - 1)
    % only load variables to append to save mamory usage
    processed = load(sprintf('%s/processed_qsos_multi_lls_occam_lyseries_variance_kim_lyb_a03_dr14q_%d-%d', ...
         processed_directory(release), ...
         i * num_quasars_split, (i + 1) * num_quasars_split ), ...
         variables_to_append{:});
    
    qso_start = i * num_quasars_split + 1;
    qso_end   = (i + 1) * num_quasars_split;
    
    % hardcoded to append everything
    % variables_to_append
    min_z_dlas(qso_start:qso_end, 1)             = processed.min_z_dlas(:, 1);
    max_z_dlas(qso_start:qso_end, 1)             = processed.max_z_dlas(:, 1);
    log_priors_no_dla(qso_start:qso_end, 1)      = processed.log_priors_no_dla(:, 1);
    log_priors_dla(qso_start:qso_end, :)         = processed.log_priors_dla(:, :);
    log_likelihoods_no_dla(qso_start:qso_end, 1) = processed.log_likelihoods_no_dla(:, 1);
    sample_log_likelihoods_dla(qso_start:qso_end, :, :) = processed.sample_log_likelihoods_dla(:, :, :);
    base_sample_inds(qso_start:qso_end, :, :)    = processed.base_sample_inds(:, :, :);
    log_likelihoods_dla(qso_start:qso_end, :)    = processed.log_likelihoods_dla(:, :);
    log_posteriors_no_dla(qso_start:qso_end, 1)  = processed.log_posteriors_no_dla(:, 1);
    log_posteriors_dla(qso_start:qso_end, :)     = processed.log_posteriors_dla(:, :);

    % initialize lls results
    log_likelihoods_lls(qso_start:qso_end, 1) = processed.log_likelihoods_lls(:, 1);
    log_posteriors_lls(qso_start:qso_end, 1)  = processed.log_posteriors_lls(:, 1);
    log_priors_lls(qso_start:qso_end, 1)      = processed.log_priors_lls(:, 1);
    sample_log_likelihoods_lls(qso_start:qso_end, :) = processed.sample_log_likelihoods_lls(:, :);

    % save maps: add the initilizations of MAP values
    % N * (1~k models) * (1~k MAP dlas)
    MAP_z_dlas(qso_start:qso_end, :, :)   = processed.MAP_z_dlas(:, :, :);
    MAP_log_nhis(qso_start:qso_end, :, :) = processed.MAP_log_nhis(:, :, :);

    all_exceptions(qso_start:qso_end, 1)  = processed.all_exceptions(:, 1);

    % posteriors results which are not in init of process script
    model_posteriors(qso_start:qso_end, :) = processed.model_posteriors(:, :);
    p_no_dlas(qso_start:qso_end, 1)        = processed.model_posteriors(:, 1);
    p_dlas(qso_start:qso_end, 1)           = processed.p_dlas(:, 1);
    p_lls(qso_start:qso_end, 1)            = processed.p_lls(:, 1);

    clear processed    
end

% ad-hocly add the final piece of chunck
processed = load(sprintf('%s/processed_qsos_multi_lls_occam_lyseries_variance_kim_lyb_a03_dr14q_%d-%d', ...
    processed_directory(release), ...
    num_files * num_quasars_split, ...
    num_files * num_quasars_split + num_final_chunk), ...
    variables_to_append{:});

qso_start = num_files * num_quasars_split + 1;
qso_end   = num_files * num_quasars_split + num_final_chunk;

% hardcoded to append everything
% variables_to_append
min_z_dlas(qso_start:qso_end, 1)             = processed.min_z_dlas(:, 1);
max_z_dlas(qso_start:qso_end, 1)             = processed.max_z_dlas(:, 1);
log_priors_no_dla(qso_start:qso_end, 1)      = processed.log_priors_no_dla(:, 1);
log_priors_dla(qso_start:qso_end, :)         = processed.log_priors_dla(:, :);
log_likelihoods_no_dla(qso_start:qso_end, 1) = processed.log_likelihoods_no_dla(:, 1);
sample_log_likelihoods_dla(qso_start:qso_end, :, :) = processed.sample_log_likelihoods_dla(:, :, :);
base_sample_inds(qso_start:qso_end, :, :)    = processed.base_sample_inds(:, :, :);
log_likelihoods_dla(qso_start:qso_end, :)    = processed.log_likelihoods_dla(:, :);
log_posteriors_no_dla(qso_start:qso_end, 1)  = processed.log_posteriors_no_dla(:, 1);
log_posteriors_dla(qso_start:qso_end, :)     = processed.log_posteriors_dla(:, :);

% initialize lls results
log_likelihoods_lls(qso_start:qso_end, 1) = processed.log_likelihoods_lls(:, 1);
log_posteriors_lls(qso_start:qso_end, 1)  = processed.log_posteriors_lls(:, 1);
log_priors_lls(qso_start:qso_end, 1)      = processed.log_priors_lls(:, 1);
sample_log_likelihoods_lls(qso_start:qso_end, :) = processed.sample_log_likelihoods_lls(:, :);

% save maps: add the initilizations of MAP values
% N * (1~k models) * (1~k MAP dlas)
MAP_z_dlas(qso_start:qso_end, :, :)   = processed.MAP_z_dlas(:, :, :);
MAP_log_nhis(qso_start:qso_end, :, :) = processed.MAP_log_nhis(:, :, :);

all_exceptions(qso_start:qso_end, 1)  = processed.all_exceptions(:, 1);

% posteriors results which are not in init of process script
model_posteriors(qso_start:qso_end, :) = processed.model_posteriors(:, :);
p_no_dlas(qso_start:qso_end, 1)        = processed.model_posteriors(:, 1);
p_dlas(qso_start:qso_end, 1)           = processed.p_dlas(:, 1);
p_lls(qso_start:qso_end, 1)            = processed.p_lls(:, 1);

clear processed

fprintf_debug('Number of isnan : %d; number of all_exceptions : %d', ...
    sum(isnan(log_likelihoods_no_dla)), sum(~isnan(all_exceptions)));

% save results
% exclude base_sample_inds and sample_log_likelihoods_dla
variables_to_save = {'training_release', 'training_set_name', ...
                     'dla_catalog_name', 'prior_ind', 'release', ...
                     'test_set_name', 'prior_z_qso_increase', 'k', ...
                     'normalization_min_lambda', 'normalization_max_lambda', ...
                     'min_z_cut', 'max_z_cut', 'num_dla_samples', ...
                     'num_lines', 'min_z_dlas', 'max_z_dlas', ...
                     'sample_log_likelihoods_dla', 'base_sample_inds', ...
                     'log_priors_no_dla', 'log_priors_dla', 'log_priors_lls', ...
                     'log_likelihoods_no_dla', 'MAP_z_dlas', 'MAP_log_nhis', ...
                     'log_likelihoods_dla', 'log_likelihoods_lls', ...
                     'log_posteriors_no_dla', 'log_posteriors_dla', 'log_posteriors_lls', ...
                     'model_posteriors', 'p_no_dlas', 'p_dlas', 'p_lls', ...
                     'all_exceptions', 'sample_log_likelihoods_lls', 'test_ind'};

filename = sprintf('%s/processed_qsos_multi_lls_occam_lyseries_variance_kim_lyb_a03_%s', ...
                    processed_directory(release), ...
                    test_set_name);

save(filename, variables_to_save{:}, '-v7.3');
