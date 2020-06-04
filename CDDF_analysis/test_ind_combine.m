% re-order the index of the partial catalog to fit Jacob's test_ind

set_parameters;
release           = 'dr12q';
test_set_name     = 'dr12q';

% the size of sub and sup depends on this
DLA_cut = 20.3;

% load redshifts from catalog to process
catalog    = load(sprintf('%s/zqso_only_catalog', processed_directory(release)));
catalog_jf = load(sprintf('%s/catalog_jfaub', processed_directory(release)));

load(sprintf('mf2jf_ind_file'))

test_ind    = catalog.filter_flags    == 0;
test_ind_jf = catalog_jf.filter_flags == 0;

thing_ids    = catalog.thing_ids(test_ind);
thing_ids_jf = catalog.thing_ids(test_ind_jf);

% only later half
thing_ids    = thing_ids(80001:end);
thing_ids_jf = thing_ids_jf(80001:end);

assert( all(thing_ids(mf2jf_ind) == thing_ids_jf(jf2mf_ind)) )

% acquire the size of Jacob's later half processed file
[num_quasars, ~] = size(thing_ids_jf);

% hard code these init vars since this is a backup script to when sbatch_reunion
% is not working anyway
variables_to_load = {'training_release', 'training_set_name', ...
    'dla_catalog_name', 'release', ...
    'test_set_name', 'test_ind', 'prior_z_qso_increase', ...
    'max_z_cut', 'num_lines', 'all_thing_ids', ...
    'sample_log_posteriors_dla_sub'}; % load this to verify the size

% this is just to get the variables we do not need to append
% to fit MATLAB's convention, you start with +1
quasar_start_ind = 80000 + 1;
quasar_end_ind   = 80000 + 50 + 1;

load(sprintf('%s/processed_qsos_zqsos_sbird_%s-%s_%d-%d_norm_%d-%d',             ...
     processed_directory(release), ...
     test_set_name, optTag, ...
     quasar_start_ind, quasar_end_ind, ...
     normalization_min_lambda, normalization_max_lambda), ...
     variables_to_load{:});

%%%%%%%%%%%%%%%%%%%%%%%%%% replacement happended %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% replace some variables to Jacob's variables based on Jacob's test_ind
test_ind = test_ind_jf;
all_thing_ids = thing_ids_jf;

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
model_posteriors = nan(num_quasars, 3);
p_no_dlas        = nan(num_quasars, 1);
p_dlas           = nan(num_quasars, 1);

%%%%%%%%%%%%%%%%%%%%%%%%%% replacement happended %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

% hardcoded to append everything
% append my variables based on Jacob's test_ind
min_z_dlas(jf2mf_ind, :)             = processed.min_z_dlas(mf2jf_ind, :);
max_z_dlas(jf2mf_ind, :)             = processed.max_z_dlas(mf2jf_ind, :);

sample_log_posteriors_no_dla(jf2mf_ind, :)  = processed.sample_log_posteriors_no_dla(mf2jf_ind, :);
sample_log_posteriors_dla(jf2mf_ind, :)     = processed.sample_log_posteriors_dla(mf2jf_ind, :);

sample_log_posteriors_dla_sub(jf2mf_ind, :) = processed.sample_log_posteriors_dla_sub(mf2jf_ind, :);
sample_log_posteriors_dla_sup(jf2mf_ind, :) = processed.sample_log_posteriors_dla_sup(mf2jf_ind, :);

log_posteriors_no_dla(jf2mf_ind, 1)  = processed.log_posteriors_no_dla(mf2jf_ind, 1);
log_posteriors_dla(jf2mf_ind, 1)     = processed.log_posteriors_dla(mf2jf_ind, 1);
log_posteriors_dla_sub(jf2mf_ind, 1) = processed.log_posteriors_dla_sub(mf2jf_ind, 1);
log_posteriors_dla_sup(jf2mf_ind, 1) = processed.log_posteriors_dla_sup(mf2jf_ind, 1);

z_true(jf2mf_ind, 1)          = processed.z_true(mf2jf_ind, 1);
dla_true(jf2mf_ind, 1)        = processed.dla_true(mf2jf_ind, 1);
z_map(jf2mf_ind, 1)           = processed.z_map(mf2jf_ind, 1);
z_dla_map(jf2mf_ind, 1)       = processed.z_dla_map(mf2jf_ind, 1);
n_hi_map(jf2mf_ind, 1)        = processed.n_hi_map(mf2jf_ind, 1);
log_nhi_map(jf2mf_ind, 1)     = processed.log_nhi_map(mf2jf_ind, 1);
signal_to_noise(jf2mf_ind, 1) = processed.signal_to_noise(mf2jf_ind, 1);

% posteriors results which are not in init of process script
model_posteriors(jf2mf_ind, :) = processed.model_posteriors(mf2jf_ind, :);
p_no_dlas(jf2mf_ind, 1)        = processed.p_no_dlas(mf2jf_ind, 1);
p_dlas(jf2mf_ind, 1)           = processed.p_dlas(mf2jf_ind, 1);

clear processed

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

filename = sprintf('%s/processed_qsos_zqsos_jacob_%s-%s_80001-%d_norm_%d-%d', ...
                    processed_directory(release), ...
                    test_set_name, optTag, ...
                    80000 + num_quasars, ...
                    normalization_min_lambda, normalization_max_lambda);

save(filename, variables_to_save{:}, '-v7.3');
