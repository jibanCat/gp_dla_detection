% process_qsos: run DLA detection algorithm on specified objects
% 
% Apr 8, 2020: add all Lyman series to the effective optical depth
%   effective_optical_depth := ∑ τ fi1 λi1 / ( f21 λ21 ) * ( 1 + z_i1 )^β
%  where 
%   1 + z_i1 =  λobs / λ_i1 = λ_lya / λ_i1 *  (1 + z_a)
% Dec 25, 2019: add Lyman series to the noise variance training
%   s(z)     = 1 - exp(-effective_optical_depth) + c_0 
% the mean values of Kim's effective optical depth
% Apr 28: add occams razor for penalising the missing pixels,
%   this factor is tuned to affect log likelihood in a range +- 500,
%   this value could be effective to penalise every likelihoods for zQSO > zCIV
%   the current implemetation is:
%     likelihood - occams_factor * (1 - lambda_observed / (max_lambda - min_lambda) )
%   and occams_factor is a tunable hyperparameter
% 
% May 11: out-of-range data penalty,
%   adding additional log-likelihoods to the null model log likelihood,
%   this additional log-likelihoods are:
%     log N(y_bluewards; bluewards_mu, diag(V_bluewards) + bluewards_sigma^2 )
%     log N(y_redwards;  redwards_mu,  diag(V_redwards)  + redwards_sigma^2 )
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

% load QSO model from training release
variables_to_load = {'rest_wavelengths', 'mu', 'M', 'log_omega', ...
    'log_c_0', 'log_tau_0', 'log_beta', ...
    'bluewards_mu', 'bluewards_sigma', ...
    'redwards_mu', 'redwards_sigma'};
load(sprintf('%s/learned_model_continuum_%s_norm_%d-%d_z_%d-%d',           ...
     processed_directory(training_release), ...
     training_set_name, ...
     normalization_min_lambda, normalization_max_lambda, ...
     z_qso_training_min_cut, z_qso_training_max_cut),  ...
     variables_to_load{:});

% load redshifts from catalog to process
catalog = load(sprintf('%s/catalog', processed_directory(release)));

% load preprocessed QSOs
variables_to_load = {'all_wavelengths', 'all_flux', 'all_noise_variance', ...
    'all_pixel_mask'};
load(sprintf('%s/preloaded_qsos', processed_directory(release)), ...
    variables_to_load{:});

% enable processing specific QSOs via setting to_test_ind
if (ischar(test_ind))
    test_ind = eval(test_ind);
end

all_wavelengths    =    all_wavelengths(test_ind);
all_flux           =           all_flux(test_ind);
all_noise_variance = all_noise_variance(test_ind);
all_pixel_mask     =     all_pixel_mask(test_ind);
all_thing_ids      =   catalog.thing_ids(test_ind);

z_qsos = catalog.z_qsos(test_ind);

num_quasars = numel(z_qsos);
if exist('qso_ind', 'var') == 0
    qso_ind = 1:1:floor(num_quasars/100);
end
num_quasars = numel(qso_ind);

%load('./test/M.mat');
% preprocess model interpolants
mu_interpolator = ...
    griddedInterpolant(rest_wavelengths,        mu,        'linear');
M_interpolator = ...
    griddedInterpolant({rest_wavelengths, 1:k}, M,         'linear');
log_omega_interpolator = ...
    griddedInterpolant(rest_wavelengths,        log_omega, 'linear');

% initialize results
sample_log_posteriors         = nan(num_quasars, num_continuum_samples);
log_posteriors                = nan(num_quasars, 1);
redden_map                    = nan(num_quasars, 1);
signal_to_noise               = nan(num_quasars, 1);

% pre-allocate the predicted continuum: only Hygroden line region
predicted_continuum = cell(num_quasars, 1);
debug_predicted_continuum = cell(length(sample_offset_reddens), 1);

c_0   = exp(log_c_0);
tau_0 = exp(log_tau_0);
beta  = exp(log_beta);

% this is just an array allow you to select a range
% of quasars to run
quasar_ind = 1;
% I prefer to split qso_ind into pieces to run. So checkpoint will interrupt the workflow
if exist('qso_ind', 'var') == 0
    try
        load(['./checkpointing/curDLA_', optTag, '.mat']); %checkmarking code
    catch ME
        0;
    end
else
    q_ind_start = quasar_ind;
end

% catch the exceptions
all_exceptions = false(num_quasars, 1);
all_posdeferrors = zeros(num_quasars, 1);

% reddening samples : normal rand number with mean 0 
rng('default');
sample_offset_reddens = normrnd(0, 2, [1, num_continuum_samples]);

for quasar_ind = q_ind_start:num_quasars %quasar list
    tic;
    quasar_num = qso_ind(quasar_ind);
    
    fprintf('processing quasar %i/%i (z_QSO = %0.4f) ...', ...
        quasar_ind, num_quasars, z_qsos(quasar_num));

    z_qso = z_qsos(quasar_num);

    % move these outside the parfor to avoid constantly querying these large arrays
    this_wavelengths    =    all_wavelengths{quasar_num};
    this_flux           =           all_flux{quasar_num};
    this_noise_variance = all_noise_variance{quasar_num};
    this_pixel_mask     =     all_pixel_mask{quasar_num};

    % Test: see if this spec is empty; this error handling line be outside parfor
    % would avoid running lots of empty spec in parallel workers
    if all(size(this_wavelengths) == [0 0])
        all_exceptions(quasar_ind, 1) = 1;
        continue;
    end

    % convert to QSO rest frame
    this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso);

    % normalizing here
    % this branch uses the zestimation version of the code, so no pre-normalization done.
    % this also makes sense for fitting continuum. We shouldn't have a pre-normalized
    % spec for continuum fitting.
    ind = (this_rest_wavelengths >= normalization_min_lambda) & ...
        (this_rest_wavelengths <= normalization_max_lambda);

    this_median         = nanmedian(this_flux(ind));
    this_flux           = this_flux / this_median;
    this_noise_variance = this_noise_variance / this_median .^ 2;

    % only selected the rest-frame range we've modelled
    ind = (this_rest_wavelengths >= min_lambda) & ...
        (this_rest_wavelengths <= max_lambda);

    % update the observation data with the mask provided by SDSS
    ind = ind & (~this_pixel_mask);

    this_wavelengths      =      this_wavelengths(ind);
    this_rest_wavelengths = this_rest_wavelengths(ind);
    this_flux             =             this_flux(ind);
    this_noise_variance   =   this_noise_variance(ind);

    % interpolate model onto given wavelengths
    this_mu = mu_interpolator( this_rest_wavelengths);
    this_M  =  M_interpolator({this_rest_wavelengths, 1:k});

    this_log_omega = log_omega_interpolator(this_rest_wavelengths);
    this_omega2 = exp(2 * this_log_omega);

    % set Lyseries absorber redshift for mean-flux suppression
    % apply the lya_absorption after the interpolation because NaN will appear in this_mu
    total_optical_depth = effective_optical_depth(this_wavelengths, ...
        prev_beta, prev_tau_0, z_qso, ...
        all_transition_wavelengths, all_oscillator_strengths, ...
        num_forest_lines);

    % total absorption effect of Lyseries absorption on the mean-flux
    lya_absorption = exp(- sum(total_optical_depth, 2) );

    this_mu_mf = this_mu .* lya_absorption;
    this_M_mf  = this_M  .* lya_absorption;

    % set another Lysieres absorber redshift to use in coveriance
    lya_optical_depth = effective_optical_depth(this_wavelengths, ...
        beta, tau_0, z_qso, ...
        all_transition_wavelengths, all_oscillator_strengths, ...
        num_forest_lines);

    this_scaling_factor = 1 - exp( -sum(lya_optical_depth, 2) ) + c_0;

    % this is the omega included the Lyseries
    this_omega2 = this_omega2 .* this_scaling_factor.^2;

    % re-adjust (K + Ω) to the level of μ .* exp( -optical_depth ) = μ .* a_lya
    % now the null model likelihood is:
    % p(y | λ, zqso, v, ω, M_nodla) = N(y; μ .* a_lya, A_lya (K + Ω) A_lya + V)
    this_omega2_mf = this_omega2 .* lya_absorption.^2;

    % record posdef error;
    % if it only happens for some samples not all of the samples, I would prefer
    % to think it is due to the noise_variace of the incomplete data combine with
    % the K causing the Covariance behaving weirdly.
    this_posdeferror = false(1, num_dla_samples);

    % this_sample_log_priors(1, i) = ...
    %     log(                   this_num_dlas) - log(this_num_quasars);

    parfor i = 1:num_continuum_samples
        this_redded_mu = redden_mu(this_mu, this_rest_wavelengths, ...
            sample_offset_reddens(i), ...
            normalization_min_lambda, normalization_max_lambda);

        % baseline: probability of no DLA model
        % The error handler to deal with Postive definite errors sometime happen
        try
            this_sample_log_likelihoods_no_dla(1, i) = ...
                log_mvnpdf_low_rank(this_flux, this_redded_mu, this_M, ...
                this_omega2 + this_noise_variance) ...

        catch ME
            if (strcmp(ME.identifier, 'MATLAB:posdef'))
                this_posdeferror(1, i) = true;
                fprintf('(QSO %d, Sample %d): Matrix must be positive definite. We skip this sample but you need to be careful about this spectrum', quasar_num, i)
                continue
            end
                rethrow(ME)
        end

        % save for debug : one full sample of continuum fitting
        if mod(quasar_ind, 100) == 0
            % select y1, y2, mu1, mu2, M1, M2, d1, d2
            % 1: Hydrogen absorption region
            % 2: Metal-line region
            ind_1    = this_rest_wavelengths <= lya_wavelength;
            y2       =  this_flux(~ind_1);
            this_mu1 =    this_redded_mu( ind_1);
            this_mu2 =    this_redded_mu(~ind_1);
            this_M1  =     this_M( ind_1, :);
            this_M2  =     this_M(~ind_1, :);
            d1       = this_noise_variance( ind_1) + this_omega2( ind_1);
            d2       = this_noise_variance(~ind_1) + this_omega2(~ind_1);

            [mu1, Sigma11] = conditional_mvnpdf_low_rank(y2, ...
                this_mu1, this_mu2, this_M1, this_M2, d1, d2);

            debug_predicted_continuum{i} = mu1;
            debug_this_rest_wavelengths = this_rest_wavelengths;
            debug_this_flux             = this_flux;
        end
    end

    % use nanmax to avoid NaN potentially in the samples
    % not sure whether the z code has many NaNs in array; the multi-dla codes have many NaNs
    max_log_likelihood = nanmax(sample_log_posteriors(quasar_ind, :));

    sample_probabilities = ...
        exp(sample_log_posteriors(quasar_ind, :) - ...
            max_log_likelihood);

    [~, I] = nanmax(sample_probabilities);

    % make MAP prediction of continuum
    redden_map(quasar_ind) = sample_offset_reddens(I);                                  %MAP estimate
    this_redded_mu = redden_mu(this_mu, this_rest_wavelengths, ...
        sample_offset_reddens(I), ...
        normalization_min_lambda, normalization_max_lambda);
    % select y1, y2, mu1, mu2, M1, M2, d1, d2
    % 1: Hydrogen absorption region
    % 2: Metal-line region
    ind_1    = this_rest_wavelengths <= lya_wavelength;
    y2       =  this_flux(~ind_1);
    this_mu1 =    this_redded_mu( ind_1);
    this_mu2 =    this_redded_mu(~ind_1);
    this_M1  =     this_M( ind_1, :);
    this_M2  =     this_M(~ind_1, :);
    d1       = this_noise_variance( ind_1) + this_omega2( ind_1);
    d2       = this_noise_variance(~ind_1) + this_omega2(~ind_1);

    [mu1, Sigma11] = conditional_mvnpdf_low_rank(y2, ...
        this_mu1, this_mu2, this_M1, this_M2, d1, d2);

    predicted_continuum{quasar_ind} = mu1;
    log_posteriors(quasar_ind) = log(nanmean(sample_probabilities)) + max_log_likelihood;
    fprintf(' took %0.3fs.\n', toc);
    
    % z-estimation checking printing at runtime
    if mod(quasar_ind, 1) == 0
        t = toc;
        fprintf('Done QSO %i of %i in %0.3f s. z_QSO = %0.4f, I=%d redden_map=%0.4f dif = %.04f\n', ...
            quasar_ind, num_quasars, t, z_qsos(quasar_num), I, redden_map(quasar_ind) );
    end    

    % record posdef error;
    % count number of posdef errors; if it is == num_dla_samples, then we have troubles.
    all_posdeferrors(quasar_ind, 1) = sum(this_posdeferror);
end


% save results
variables_to_save = {'training_release', 'training_set_name', 'release', ...
    'test_set_name', 'test_ind', 'predicted_continuum', ...
    'debug_predicted_continuum', 'debug_this_flux', 'debug_this_rest_wavelengths', ...
    'sample_log_posteriors', 'log_posteriors', 'redden_map', ...
    'all_thing_ids', 'all_posdeferrors', 'all_exceptions', 'qso_ind', ...
    };

filename = sprintf('%s/processed_qsos_continuum_%s_%d-%d', ...
    processed_directory(release), ...
    test_set_name, ...
    qso_ind(1), qso_ind(1) + numel(qso_ind));

save(filename, variables_to_save{:}, '-v7.3');
