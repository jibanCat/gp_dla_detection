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

% the preload_qsos should fliter out empty spectra;
% this line is to prevent there is any empty spectra
% in preloaded_qsos.mat for some reason
is_empty             = false(num_quasars, 1);

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

  %normalizing here
  ind = (this_rest_wavelengths >= normalization_min_lambda) & ...
        (this_rest_wavelengths <= normalization_max_lambda) & ...
        (~this_pixel_mask);

  this_median = nanmedian(this_flux(ind));
  rest_fluxes(i, :) = rest_fluxes(i, :) / this_median;

  rest_noise_variances(i, :) = ...
      interp1(this_rest_wavelengths, this_noise_variance, rest_wavelengths);
  rest_noise_variances(i, :) = rest_noise_variances(i, :) / this_median .^ 2;
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
for i = 1:num_quasars
  for j = 1:num_forest_lines
    all_lyman_1pzs(j, i, ind(i, :))  = nan;
  end
end

% reverse the rest_fluxes back to the fluxes before encountering Lyα forest
prev_tau_0 = 0.0023; % Kim et al. (2007) priors
prev_beta  = 3.65;

rest_fluxes_div_exp1pz      = nan(num_quasars, num_rest_pixels);
rest_noise_variances_exp1pz = nan(num_quasars, num_rest_pixels);

for i = 1:num_quasars
  % compute the total optical depth from all Lyman series members
  % Apr 8: not using NaN here anymore due to range beyond Lya will all be NaNs
  total_optical_depth = zeros(num_forest_lines, num_rest_pixels);

  for j = 1:num_forest_lines
    % calculate the oscillator strengths for Lyman series
    this_tau_0 = prev_tau_0 * ...
      all_oscillator_strengths(j)   / lya_oscillator_strength * ...
      all_transition_wavelengths(j) / lya_wavelength;
    
    % remove the leading dimension
    this_lyman_1pzs = squeeze(all_lyman_1pzs(j, i, :))'; % (1, num_rest_pixels)

    total_optical_depth(j, :) = this_tau_0 .* (this_lyman_1pzs.^prev_beta);
  end

  % Apr 8: using zeros instead so not nansum here anymore
  % beyond lya, absorption fcn shoud be unity
  lya_absorption = exp(- sum(total_optical_depth, 1) );

  % We have to reverse the effect of Lyα for both mean-flux and observational noise
  rest_fluxes_div_exp1pz(i, :)      = rest_fluxes(i, :) ./ lya_absorption;
  rest_noise_variances_exp1pz(i, :) = rest_noise_variances(i, :) ./ (lya_absorption.^2);
end

clear('all_lyman_1pzs');

% Filter out spectra which have too many NaN pixels
ind = sum(isnan(rest_fluxes),2) < num_rest_pixels-min_num_pixels;
fprintf("Filtering %g quasars for NaN\n", length(rest_fluxes) - nnz(ind));
rest_fluxes = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);
% Check for columns which contain only NaN on either end.
nancolfrac = sum(isnan(rest_fluxes_div_exp1pz), 1) / nnz(ind);
fprintf("Columns with nan > 0.9: ");
max(find(nancolfrac > 0.9))

% find empirical mean vector and center data
mu = nanmean(rest_fluxes_div_exp1pz);
centered_rest_fluxes = bsxfun(@minus, rest_fluxes_div_exp1pz, mu);
clear('rest_fluxes', 'rest_fluxes_div_exp1pz');

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
