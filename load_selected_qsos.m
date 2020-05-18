% load_selected_qsos.m : load only one quasar for testing the model

% select the thing_ids you want to test the model
selected_thing_ids = [267231131, 467627896, 290515842, 430536819, 257809711, ...
    341394135, 78109198, 34953145, 431593999, 526683132, 84523031, 84621480, ...
    268514930, 66692618, 374107809];

% prior settings
% specify the learned quasar model to use
training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';

% specify the spectra to process
release       = 'dr12q';
test_set_name = 'dr12q';
test_ind      = '(catalog.filter_flags == 0)';

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
    'log_c_0', 'log_tau_0', 'log_beta'};
load(sprintf('%s/learned_model_outdata_light_%s_norm_%d-%d',           ...
     processed_directory(training_release), ...
     training_set_name, ...
     normalization_min_lambda, normalization_max_lambda),  ...
     variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/catalog', processed_directory(release)));

% enable processing specific QSOs via setting to_test_ind
if (ischar(test_ind))
  test_ind = eval(test_ind);
end

z_qsos    = catalog.z_qsos(test_ind);
thing_ids = catalog.thing_ids(test_ind);

% select the quasar_ids we like to work on
[vals, selected_quasar_inds]= intersect(thing_ids, selected_thing_ids, 'stable');
selected_thing_ids = vals; % update the vals since the ordering would change

% index mapping from pre- test_ind to test_ind
real_index          = find(test_ind);
selected_real_index = real_index(selected_quasar_inds);

% load preprocessed QSOs
preloaded_qsos = matfile(sprintf('%s/preloaded_qsos', processed_directory(release)));

all_wavelengths    = preloaded_qsos.all_wavelengths;
all_wavelengths    = all_wavelengths(selected_real_index, :);
all_flux           = preloaded_qsos.all_flux;
all_flux           = all_flux(selected_real_index, :);
all_noise_variance =  preloaded_qsos.all_noise_variance;
all_noise_variance = all_noise_variance(selected_real_index, :);
all_pixel_mask     = preloaded_qsos.all_pixel_mask;
all_pixel_mask     = all_pixel_mask(selected_real_index, :);

clear preloaded_qsos

% update loaded catalog values to only selected quasars
num_quasars = numel(selected_real_index); 
z_qsos      = z_qsos(selected_quasar_inds);
thing_ids   = thing_ids(selected_quasar_inds);

assert( all(thing_ids == selected_thing_ids) )

% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');
log_omega_interpolator = ...
    griddedInterpolant(rest_wavelengths,        log_omega, 'linear');

c_0   = exp(log_c_0);
tau_0 = exp(log_tau_0);
beta  = exp(log_beta);

% pre-trained Lyα absorption parameters
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

num_forest_lines = 31;

