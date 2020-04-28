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
num_rest_pixels  = numel(rest_wavelengths);

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

  fprintf('processing quasar %i with lambda_size = %i %i ...\n', i, size(this_wavelengths))
  
  if all(size(this_wavelengths) == [0 0])
    is_empty(i, 1) = 1;
    continue;
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
  rest_noise_variances(i, :) = rest_noise_variances(i, :) / this_median .^ 2;
end
clear('all_wavelengths', 'all_flux', 'all_noise_variance', 'all_pixel_mask');

% filter out empty spectra
% note: if you've done this in preload_qsos then skip these lines
z_qsos               = z_qsos(~is_empty);
rest_fluxes          = rest_fluxes(~is_empty, :);
rest_noise_variances = rest_noise_variances(~is_empty, :);

% update num_quasars in consideration
num_quasars = numel(z_qsos);

fprintf('Get rid of empty spectra, num_quasars = %i\n', num_quasars);

% Filter out spectra with redshifts outside the training region
ind = (z_qsos > z_qso_training_min_cut) & (z_qsos < z_qso_training_max_cut);

fprintf("Filtering %g quasars for redshift\n", length(rest_fluxes) - nnz(ind));

rest_fluxes          = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);

% mask noisy pixels
ind = (rest_noise_variances > max_noise_variance);

fprintf("Masking %g of pixels\n", nnz(ind) * 1 ./ numel(ind));

rest_fluxes(ind)          = nan;
rest_noise_variances(ind) = nan;

% Filter out spectra which have too many NaN pixels
ind = sum(isnan(rest_fluxes),2) < num_rest_pixels-min_num_pixels;

fprintf("Filtering %g quasars for NaN\n", length(rest_fluxes) - nnz(ind));

rest_fluxes          = rest_fluxes(ind, :);
rest_noise_variances = rest_noise_variances(ind,:);

% Check for columns which contain only NaN on either end.
nancolfrac = sum(isnan(rest_fluxes), 1) / nnz(ind);

fprintf("Columns with nan > 0.9: ");

max(find(nancolfrac > 0.9))

% find the power-law fit indexs for each rest_fluxes
[num_quasars, ~] = size(rest_fluxes);

all_ps = nan(num_quasars, 2);

for i = 1:num_quasars
  this_rest_flux = rest_fluxes(i, :);

  % pick up the emission free regions
  ind = (1300 < rest_wavelengths) & (rest_wavelengths < 1350);
  ind = ind | (1425 < rest_wavelengths) & (rest_wavelengths < 1500);
  ind = ind | (1650 < rest_wavelengths) & (rest_wavelengths < 1750);
  ind = ind | (2000 < rest_wavelengths) & (rest_wavelengths < 2200);

  this_rest_flux           = this_rest_flux(ind);
  this_rest_noise_variance = rest_noise_variances(ind);
  this_rest_wavelengths    = rest_wavelengths(ind);

  % to avoid the problem for negative flux
  neg_ind = (this_rest_flux <= 0);
  this_rest_flux(neg_ind) = nanmedian(this_rest_flux);
  fprintf('Number of negative flux pixels: %d', sum(neg_ind))

  % do the polyfit(1-) in the log space
  %   log( flux ) = b log( lambda ) + log( a )
  % which means the power-law fit
  %   flux = a lambda^b
  % we are seeking a different gamma param population
  p = polyfit(log(rest_wavelengths), log(this_rest_flux), 1);

  b = p(1); % power-law index
  a = p(2); % scalar factor, should be simular

  fprintf('- The multiplication factor a : %d', a)
  fprintf('- The power-law index b       : %d', b)

  all_ps(i, 1) = b;
  all_ps(i, 2) = a;
end

% negative power-law index
ind = all_ps(:, 1) < 0;
fprintf('Totoal number of positive power-law trend QSOs (possibily extinction) : %d', sum(~ind))

% find empirical mean vector and center data
% model1: model with positive power-law (positive extinction)
mu1 = nanmean(rest_fluxes(~ind));
centered_rest_fluxes1 = bsxfun(@minus, rest_fluxes(~ind), mu1);

% model2: model with negative power-law
mu2 = nanmean(rest_fluxes(ind));
centered_rest_fluxes2 = bsxfun(@minus, rest_fluxes(ind),  mu2);

clear('rest_fluxes');

% % % small fix to the data fit into the pca:
% % % make the NaNs to the medians of a given row
% % % rememeber not to inject this into the actual
% % % joint likelihood maximisation
% % pca_centered_rest_flux = centered_rest_fluxes;

% % [num_quasars, ~] = size(pca_centered_rest_flux);

% % for i = 1:num_quasars
% %   this_pca_cetnered_rest_flux = pca_centered_rest_flux(i, :);

% %   % assign median value for each row to nan
% %   ind = isnan(this_pca_cetnered_rest_flux);
  
% %   pca_centered_rest_flux(i, ind) = nanmedian(this_pca_cetnered_rest_flux);
% % end

% % get top-k PCA vectors to initialize M
% [coefficients, ~, latent] = ...
%   pca_custom(centered_rest_fluxes, ...
%         'numcomponents', k, ...
%         'rows',          'pairwise');
% % initialize A to top-k PCA components of non-DLA-containing spectra
% initial_M = bsxfun(@times, coefficients(:, 1:k), sqrt(latent(1:k))');

% objective_function = @(x) objective(x, centered_rest_fluxes, rest_noise_variances);

% % maximize likelihood via L-BFGS
% [x, log_likelihood, ~, minFunc_output] = ...
%     minFunc(objective_function, initial_M, minFunc_options);

% ind = (1:(num_rest_pixels * k));
% M = reshape(x(ind), [num_rest_pixels, k]);

% variables_to_save = {'training_release', 'train_ind', 'max_noise_variance', ...
%                      'minFunc_options', 'rest_wavelengths', 'mu', ...
%                      'initial_M', 'M',  'log_likelihood', ...
%                      'minFunc_output'};

variables_to_save = {'all_ps', 'mu1', 'mu2'};


save(sprintf('%s/learned_two_model_zqso_only_model_%s',             ...
             processed_directory(training_release), ...
             training_set_name), ...
     variables_to_save{:}, '-v7.3');
