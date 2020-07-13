% learn_qso_model: fits GP to training catalog via maximum likelihood

rng('default');

% load catalog
catalog = load(sprintf('%s/catalog', processed_directory(training_release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
                     'all_pixel_mask'};
preqsos = matfile(sprintf('%s/preloaded_qsos.mat', processed_directory(training_release)));

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
num_rest_pixels  = numel(rest_wavelengths);

lya_1pzs             = nan(num_quasars, num_rest_pixels);
all_lyman_1pzs       = nan(num_forest_lines, num_quasars, num_rest_pixels);
rest_fluxes          = nan(num_quasars, num_rest_pixels);
rest_noise_variances = nan(num_quasars, num_rest_pixels);
% save these two variables to minimize the reddening
all_lya_absorptions  = nan(num_quasars, num_rest_pixels);
all_reddens          = nan(num_quasars, 1);

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

  fprintf('processing quasar %i with lambda_size = %i %i ...\n', i, size(this_wavelengths));

  if all(size(this_wavelengths) == [0 0])
    is_empty(i, 1) = 1;
    continue;
  end

  lya_1pzs(i, :) = ...
      interp1(this_rest_wavelengths, ...
              1 + (this_wavelengths - lya_wavelength) / lya_wavelength, ...
              rest_wavelengths);
  
  % this_wavelength is raw wavelength (w/t ind)
  % so we need an indicator here to comfine lya_1pzs
  % below Lyman alpha (do we need to make the indicator
  % has a lower bound at Lyman limit here?)
  % indicator = lya_1pzs(i, :) <= (1 + z_qso);
  % lya_1pzs(i, :) = lya_1pzs(i, :) .* indicator;

  % include all members in Lyman series to the forest
  for j = 1:num_forest_lines
    this_transition_wavelength = all_transition_wavelengths(j);

    all_lyman_1pzs(j, i, :) = ...
      interp1(this_rest_wavelengths, ...
              1 + (this_wavelengths - this_transition_wavelength) / this_transition_wavelength, ... 
              rest_wavelengths);

    % indicator function: z absorbers <= z_qso
    indicator = all_lyman_1pzs(j, i, :) <= (1 + z_qso);

    all_lyman_1pzs(j, i, :) = all_lyman_1pzs(j, i, :) .* indicator;
  end

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
  rest_noise_variances(i, :) = rest_noise_variances(i, :) / this_median .^ 2;  %setting up bluward/redwards of restframe txt files
end
clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');

% filter out empty spectra
% note: if you've done this in preload_qsos then skip these lines
z_qsos               = z_qsos(~is_empty);
lya_1pzs             = lya_1pzs(~is_empty, :);
rest_fluxes          = rest_fluxes(~is_empty, :);
rest_noise_variances = rest_noise_variances(~is_empty, :);
all_lyman_1pzs       = all_lyman_1pzs(:, ~is_empty, :);

fprintf('Get rid of empty spectra, num_quasars = %i\n', num_quasars);

% Filter out spectra with redshifts outside the training region
% for continuum fit, we should select a very low redshift region, e.g., 2.15 - 2.8
ind = (z_qsos > z_qso_training_min_cut) & (z_qsos < z_qso_training_max_cut);
fprintf("Filtering %g quasars for redshift\n", length(rest_fluxes) - nnz(ind));
z_qsos               = z_qsos(ind);
lya_1pzs             = lya_1pzs(ind, :);
rest_fluxes          = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);
all_lyman_1pzs       = all_lyman_1pzs(:, ind, :);

% update num_quasars in consideration
num_quasars = numel(z_qsos);

% mask noisy pixels
ind = (rest_noise_variances > max_noise_variance);
fprintf("Masking %g of pixels\n", nnz(ind) * 1 ./ numel(ind));
lya_1pzs(ind)             = nan;
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
  all_lya_absorptions(i, :) = lya_absorption;

  % We have to reverse the effect of Lyα for both mean-flux and observational noise
  rest_fluxes_div_exp1pz(i, :)      = rest_fluxes(i, :) ./ lya_absorption;
  rest_noise_variances_exp1pz(i, :) = rest_noise_variances(i, :) ./ (lya_absorption.^2);
end

clear('all_lyman_1pzs');

% Filter out spectra which have too many NaN pixels
ind = sum(isnan(rest_fluxes_div_exp1pz),2) < num_rest_pixels-min_num_pixels;

fprintf("Filtering %g quasars\n", length(rest_fluxes_div_exp1pz) - nnz(ind));

z_qsos                      = z_qsos(ind);
rest_fluxes                 = rest_fluxes(ind, :);
rest_fluxes_div_exp1pz      = rest_fluxes_div_exp1pz(ind, :);
rest_noise_variances_exp1pz = rest_noise_variances_exp1pz(ind, :);
lya_1pzs                    = lya_1pzs(ind, :);
all_lya_absorptions         = all_lya_absorptions(ind, :);

% update num_quasars in consideration
num_quasars = numel(z_qsos);

% Check for columns which contain only NaN on either end.
nancolfrac = sum(isnan(rest_fluxes_div_exp1pz), 1) / nnz(ind);
fprintf("Columns with nan > 0.9: ");

max(find(nancolfrac > 0.9))

% find empirical mean vector and center data
mu = nanmean(rest_fluxes_div_exp1pz);
% centered_rest_fluxes = bsxfun(@minus, rest_fluxes_div_exp1pz, mu);
centered_rest_fluxes = nan(num_quasars, num_rest_pixels);

% find the reddening factor for each qso
for i = 1:num_quasars
  a0 = 0.;
  objective_fun = @(a) objective_reddening(rest_wavelengths, ...
    rest_fluxes(i, :), all_lya_absorptions(i, :), mu, ...
    a, normalization_min_lambda, normalization_max_lambda);
  redden = fminsearch(objective_fun, a0);

  all_reddens(i, 1) = redden;

  % get the ( y - mu ) with consideration of reddening
  this_mu = redden_mu(mu, rest_wavelengths, redden, ...
    normalization_min_lambda, normalization_max_lambda);
  centered_rest_fluxes(i, :) = ( rest_fluxes_div_exp1pz(i, :) - this_mu );
end

% save the fluxes and reddening for future debug
variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
                     'rest_wavelengths', 'mu', 'all_reddens', ...
                     'rest_fluxes', 'rest_fluxes_div_exp1pz', ...
                     'all_lya_absorptions', 'centered_rest_fluxes' ...
                     };

save(sprintf('%s/saved_reddening_fluxes_%s',             ...
             processed_directory(training_release), ...
             training_set_name), ...
     variables_to_save{:}, '-v7.3');
clear('rest_fluxes', 'rest_fluxes_div_exp1pz', 'all_lya_absorptions');

% small fix to the data fit into the pca:
% make the NaNs to the medians of a given row
% rememeber not to inject this into the actual
% joint likelihood maximisation
pca_centered_rest_flux = centered_rest_fluxes;

[num_quasars, ~] = size(pca_centered_rest_flux);

for i = 1:num_quasars
  this_pca_centered_rest_flux = pca_centered_rest_flux(i, :);

  % assign median value for each row to nan
  ind = isnan(this_pca_centered_rest_flux);

  pca_centered_rest_flux(i, ind) = nanmedian(this_pca_centered_rest_flux);
end

% get top-k PCA vectors to initialize M
[coefficients, ~, latent] = ...
  pca(pca_centered_rest_flux, ...
        'numcomponents', k, ...
        'rows',          'complete');

objective_function = @(x) objective(x, centered_rest_fluxes, lya_1pzs, ...
        rest_noise_variances_exp1pz, num_forest_lines, all_transition_wavelengths, ...
        all_oscillator_strengths, z_qsos);

% initialize A to top-k PCA components of non-DLA-containing spectra
initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

% initialize log omega to log of elementwise sample standard deviation
initial_log_omega = log(nanstd(centered_rest_fluxes));

initial_log_c_0   = log(initial_c_0);
initial_log_tau_0 = log(initial_tau_0);
initial_log_beta  = log(initial_beta);

initial_x = [initial_M(:);         ...
             initial_log_omega(:); ...
             initial_log_c_0;      ...
             initial_log_tau_0;    ...
             initial_log_beta];

% maximize likelihood via L-BFGS
[x, log_likelihood, ~, minFunc_output] = ...
    minFunc(objective_function, initial_x, minFunc_options);

ind = (1:(num_rest_pixels * k));
M = reshape(x(ind), [num_rest_pixels, k]);

ind = ((num_rest_pixels * k + 1):(num_rest_pixels * (k + 1)));
log_omega = x(ind)';

log_c_0   = x(end - 2);
log_tau_0 = x(end - 1);
log_beta  = x(end);

variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
                     'minFunc_options', 'rest_wavelengths', 'mu', ...
                     'initial_M', 'initial_log_omega', 'initial_log_c_0', ...
                     'initial_tau_0', 'initial_beta',  'M', 'log_omega', ...
                     'log_c_0', 'log_tau_0', 'log_beta', 'log_likelihood', ...
                     'minFunc_output', 'all_reddens'};

save(sprintf('%s/learned_model_continuum_%s_norm_%d-%d_z_%d-%d',             ...
             processed_directory(training_release), ...
             training_set_name, ...
             normalization_min_lambda, normalization_max_lambda, ...
             z_qso_training_min_cut, z_qso_training_max_cut), ...
     variables_to_save{:}, '-v7.3');
