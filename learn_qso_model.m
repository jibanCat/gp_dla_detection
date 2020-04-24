% learn_qso_model: fits GP to training catalog via maximum likelihood

rng('default');

% load catalog
catalog = load(sprintf('%s/zqso_only_catalog', processed_directory(training_release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
preqsos = matfile(sprintf('%s/preloaded_zqso_only_qsos.mat', processed_directory(training_release)));

% determine which spectra to use for training; allow string value for
% train_ind
if (ischar(train_ind))
  train_ind = eval(train_ind);
end

% select training vectors
all_wavelengths    =          preqsos.all_wavelengths;
all_wavelengths    =    all_wavelengths(train_ind, :);
all_flux           =                 preqsos.all_flux;
all_flux           =           all_flux(train_ind, :);
all_noise_variance =       preqsos.all_noise_variance;
all_noise_variance = all_noise_variance(train_ind, :);
all_pixel_mask     =           preqsos.all_pixel_mask;
all_pixel_mask     =     all_pixel_mask(train_ind, :);
z_qsos             =        catalog.z_qsos(train_ind);
clear preqsos

num_quasars = numel(z_qsos);

rest_wavelengths = (min_lambda:dlambda:max_lambda);
num_rest_pixels = numel(rest_wavelengths);

rest_fluxes          = nan(num_quasars, num_rest_pixels);
rest_noise_variances = nan(num_quasars, num_rest_pixels);

% interpolate quasars onto chosen rest wavelength grid
for i = 1:num_quasars
  z_qso = z_qsos(i);

  this_wavelengths    =    all_wavelengths{i}';
  this_flux           =           all_flux{i}';
  this_noise_variance = all_noise_variance{i}';
  this_pixel_mask     =     all_pixel_mask{i}';

  this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

  this_flux(this_pixel_mask)           = nan;
  this_noise_variance(this_pixel_mask) = nan;

  rest_fluxes(i, :) = ...
      interp1(this_rest_wavelengths, this_flux,           rest_wavelengths);

  rest_noise_variances(i, :) = ...
      interp1(this_rest_wavelengths, this_noise_variance, rest_wavelengths);
end
clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');

% Filter out spectra with redshifts outside the training region
ind = (z_qsos > z_qso_training_min_cut) & (z_qsos < z_qso_training_max_cut);
fprintf("Filtering %g quasars for redshift\n", length(rest_fluxes) - nnz(ind));
rest_fluxes = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);

% mask noisy pixels
ind = (rest_noise_variances > max_noise_variance);
fprintf("Masking %g of pixels\n", nnz(ind)*1./numel(ind));
rest_fluxes(ind)          = nan;
rest_noise_variances(ind) = nan;

% Filter out spectra which have too many NaN pixels
ind = sum(isnan(rest_fluxes),2) < num_rest_pixels-min_num_pixels;
fprintf("Filtering %g quasars for NaN\n", length(rest_fluxes) - nnz(ind));
rest_fluxes = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);

% Check for columns which contain only NaN on either end.
nancolfrac = sum(isnan(rest_fluxes),1)/ nnz(ind);
fprintf("Columns with nan > 0.9: ");
max(find(nancolfrac > 0.9))

% find empirical mean vector and center data
mu = nanmean(rest_fluxes);
centered_rest_fluxes = bsxfun(@minus, rest_fluxes, mu);
clear('rest_fluxes');

% get top-k PCA vectors to initialize M
[coefficients, ~, latent] = ...
    pca(centered_rest_fluxes, ...
        'numcomponents', k, ...
        'rows',          'pairwise');
% initialize A to top-k PCA components of non-DLA-containing spectra
initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

objective_function = @(x) objective(x, centered_rest_fluxes, rest_noise_variances);

% maximize likelihood via L-BFGS
[x, log_likelihood, ~, minFunc_output] = ...
    minFunc(objective_function, initial_M, minFunc_options);

ind = (1:(num_rest_pixels * k));
M = reshape(x(ind), [num_rest_pixels, k]);

variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
                     'minFunc_options', 'rest_wavelengths', 'mu', ...
                     'initial_M', 'M',  'log_likelihood', ...
                     'minFunc_output'};

save(sprintf('%s/learned_zqso_only_model_%s',             ...
             processed_directory(training_release), ...
             training_set_name), ...
     variables_to_save{:}, '-v7.3');
