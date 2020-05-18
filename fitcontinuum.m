% fitcontinuum.m : fit continuum to a specific quasar spectrum
%
% The script will be further developed into a lightweighted python package
%   to run the continuum fitting on a specific spec.
%
% fit continuum in this script is done by using the mean-flux surpression
% GP to give a GP prior over full spectrum,
%   p(y | λ, v, ω, z_qso, M) = N(y; μ, Σ)
%
% The mutlivariate normal distribution conditioned on metal region is
%   p(y1 | y2, λ, v, ω, z_qso, M) = N(y1; μ1', Σ11')
% where y2 with λ1 < 1215.17, and y1 with λ2 >= 1215.17
%                    [ Σ11    Σ12 ] 
%  μ = [μ1, μ2]; Σ = |            |
%                    [ Σ21    Σ22 ]
% The μ1 will be updated:
%  μ1' = μ1 + Σ12 Σ22^-1 (y2 - μ2)
% The Σ11 will be updated:
%  Σ11' = Σ11 - Σ12 Σ22^-1 Σ21
%
% Note: Σ = (K + Ω) + V = MM' + Ω + V 
%   with Ω and V only have diagnol elements 
% 
% Parameters:
% ----
% this_rest_wavelengths : λ
% this_flux             : y
% this_noise_variance   : v
% this_omega2           : ω
% this_mu               : μ
% this_M                : M, K = MM'
% 

% MF suppression coefficient, apply the effective optical depth
% to the GP prior if we want to fit the mean-flux
prev_tau_0 = 0.0023;
prev_beta  = 3.65;

% quasar_ind = 1;

% this line starts the continuum fitting
tic;
z_qso = z_qsos(quasar_ind);

fprintf('fitting quasar %i/%i (z_true = %0.4f) ...', ...
    quasar_ind, num_quasars, z_qso);

% load a single spec with λ, y, v, masks
this_wavelengths    =    all_wavelengths{quasar_ind};
this_flux           =           all_flux{quasar_ind};
this_noise_variance = all_noise_variance{quasar_ind};
this_pixel_mask     =     all_pixel_mask{quasar_ind};

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

% select y1, y2, mu1, mu2, M1, M2, d1, d2
% 1: Hydrogen absorption region
% 2: Metal-line region
ind_1    = this_rest_wavelengths <= lya_wavelength;
y2       =  this_flux(~ind_1);
this_mu1 =    this_mu( ind_1);
this_mu2 =    this_mu(~ind_1);
this_M1  =     this_M( ind_1, :);
this_M2  =     this_M(~ind_1, :);
d1       = this_noise_variance( ind_1) + this_omega2( ind_1);
d2       = this_noise_variance(~ind_1) + this_omega2(~ind_1);

[mu1, Sigma11] = conditional_mvnpdf_low_rank(y2, ...
    this_mu1, this_mu2, this_M1, this_M2, d1, d2);

this_continuum = cat(1, mu1, this_mu2);

fprintf(' took %0.3fs.\n', toc);
