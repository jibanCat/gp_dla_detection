% set_parameters: sets various parameters for the DLA detection
% pipeline

%flags for changes
extrapolate_subdla = 0; %0 = off, 1 = on
add_proximity_zone = 0;
integrate          = 1;
optTag = [num2str(integrate), num2str(extrapolate_subdla), num2str(add_proximity_zone)];

% physical constants
lya_wavelength = 1215.6701;                   % Lyman alpha transition wavelength  Å
lyb_wavelength = 1025.7223;                   % Lyman beta  transition wavelength  Å
lyman_limit    =  911.7633;                   % Lyman limit wavelength             Å
speed_of_light = 299792458;                   % speed of light                     m s⁻¹

% converts relative velocity in km s^-1 to redshift difference
kms_to_z = @(kms) (kms * 1000) / speed_of_light;

% utility functions for redshifting
emitted_wavelengths = ...
    @(observed_wavelengths, z) (observed_wavelengths / (1 + z));

observed_wavelengths = ...
    @(emitted_wavelengths,  z) ( emitted_wavelengths * (1 + z));

release = 'dr12q';
file_loader = @(plate, mjd, fiber_id) ...
  (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
    spectra_directory(release),                  ...
    plate,                                       ...
    plate,                                       ...
    mjd,                                         ...
    fiber_id)));

training_release  = 'dr12q';
training_set_name = 'dr9q_minus_concordance';
train_ind = ...
    [' catalog.in_dr9                     & ' ...
     '(catalog.filter_flags == 0) ' ];

test_set_name = 'dr12q';

% file loading parameters
loading_min_lambda = lya_wavelength;                % range of rest wavelengths to load  Å
loading_max_lambda = 5000;                  % This maximum is set so we include CIV.
% The maximum allowed is set so that even if the peak is redshifted off the end, the
% quasar still has data in the range

% preprocessing parameters
z_qso_cut      = 2.15;         % filter out QSOs with z less than this threshold
z_qso_training_max_cut = 5; % roughly 95% of training data occurs before this redshift; assuming for normalization purposes (move to set_parameters when pleased)
z_qso_training_min_cut = 1.5; % Ignore these quasars when training
min_num_pixels = 400;                         % minimum number of non-masked pixels

% normalization parameters
normalization_min_lambda = 1549 - 50;              % range of rest wavelengths to use   Å
normalization_max_lambda = 1549 + 50;              %   for flux normalization

% null model parameters
min_lambda         = lya_wavelength;                 % range of rest wavelengths to       Å
max_lambda         = 2840;                 %   model
dlambda            = 0.25;                 % separation of wavelength grid      Å
k                  = 20;                      % rank of non-diagonal contribution
max_noise_variance = 4^2;                     % maximum pixel noise allowed during model training

% optimization parameters
minFunc_options =               ...           % optimization options for model fitting
    struct('MaxIter',     4000, ...
           'MaxFunEvals', 8000);

num_zqso_samples     = 10000;                  % number of parameter samples

% Lyman-series array: for modelling the forests of Lyman series
num_forest_lines = 6;
all_transition_wavelengths = [ ...
    1.2156701e-05, ...
    1.0257223e-05, ...
    9.725368e-06,  ...
    9.497431e-06,  ...
    9.378035e-06,  ...
    9.307483e-06,  ...
    9.262257e-06,  ...
    9.231504e-06,  ...
    9.209631e-06,  ...
    9.193514e-06,  ...
    9.181294e-06,  ...
    9.171806e-06,  ...
    9.16429e-06,   ...
    9.15824e-06,   ...
    9.15329e-06,   ...
    9.14919e-06,   ...
    9.14576e-06,   ...
    9.14286e-06,   ...
    9.14039e-06,   ...
    9.13826e-06,   ...
    9.13641e-06,   ...
    9.13480e-06,   ...
    9.13339e-06,   ...
    9.13215e-06,   ...
    9.13104e-06,   ...
    9.13006e-06,   ...
    9.12918e-06,   ...
    9.12839e-06,   ...
    9.12768e-06,   ...
    9.12703e-06,   ...
    9.12645e-06] * 1e8; % transition wavelengths, Å 

all_oscillator_strengths = [ ...
    0.416400, ...
    0.079120, ...
    0.029000, ...
    0.013940, ...
    0.007799, ...
    0.004814, ...
    0.003183, ...
    0.002216, ...
    0.001605, ...
    0.00120,  ...
    0.000921, ...
    0.0007226,...
    0.000577, ...
    0.000469, ...
    0.000386, ...
    0.000321, ...
    0.000270, ...
    0.000230, ...
    0.000197, ...
    0.000170, ...
    0.000148, ...
    0.000129, ...
    0.000114, ...
    0.000101, ...
    0.000089, ...
    0.000080, ...
    0.000071, ...
    0.000064, ...
    0.000058, ...
    0.000053, ...
    0.000048];

% oscillator strengths
lya_oscillator_strength = 0.416400;
lyb_oscillator_strength = 0.079120;

% base directory for all data
base_directory = 'data';

% utility functions for identifying various directories
distfiles_directory = @(release) ...
    sprintf('%s/%s/distfiles', base_directory, release);

spectra_directory   = @(release) ...
    sprintf('%s/%s/spectra',   base_directory, release);

processed_directory = @(release) ...
    sprintf('%s/%s/processed', base_directory, release);

% replace with @(varargin) (fprintf(varargin{:})) to show debug statements
% fprintf_debug = @(varargin) (fprintf(varargin{:}));
fprintf_debug = @(varargin) ([]);
