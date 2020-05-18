Continuum fit for BOSS quasar spectra
==============================================

This branch uses Garnett (2017)'s GP model design and
Ho (2020)'s mean-flux suppression to learn GP model.

Data loading and catalog building procedures are the same as Garnett (2017).

Lyman-series are included in both covariance and mean vector.

> R Garnett, S Ho, S Bird, and J Schnedier. Detecting Damped Lyman-α
> Absorbers with Gaussian Processes. [arXiv:1605.04460
> [astro-ph.CO]](https://arxiv.org/abs/1605.04460),

> M-F Ho, S Bird, and R Garnett. Detecting Multiple DLAs per
> Spectrum in SDSS DR12 with Gaussian Processes. [arXiv:2003.11036
> [astro-ph.CO]](https://arxiv.org/abs/2003.11036),

including all intermediate data products including the Gaussian
process null model described therein. The provided parameters should
exactly reproduce the catalog in that work; however, you may feel free
to modify these choices as you see fit.

The pipeline has multiple stages, outlined and documented below.

Loading catalogs and downloading spectra
----------------------------------------

The first step of the process is to load the DR12Q quasar catalog and
available DLA catalogs, extract some basic data such as redshift,
coordinates, etc., and apply some basic filtering to the spectra:

* spectra with z < 2.15 are filtered
* spectra identified in a visual survey to have broad absorption line
  (BAL) features are filtered

Relevant parameters in `set_parameters` that can be tweaked if desired:

    % preprocessing parameters
    z_qso_cut      = 2.15;                        % filter out QSOs with z less than this threshold

This process proceeds in three steps, alternating between the shell
and MATLAB.

First we download the raw catalog data:

    # in shell
    cd data/scripts
    ./download_catalogs.sh

Then we load these catalogs into MATLAB:

    % in MATLAB
    set_parameters;
    build_catalogs;

The `build_catalogs` script will produce a file called `file_list` in
the `data/dr12q/spectra` directory containing relative paths to
yet-unfiltered SDSS spectra for download. The `file_list` output is
available in this repository in the same location. The next step is to
download the coadded "speclite" SDSS spectra for these observations
(warning: total download is 35 GB). The `download_spectra.sh` script
requires `wget` to be available. On OS X systems, this may be
installed easily with [Homebrew](http://brew.sh/index.html).

    # in shell
    cd data/scripts
    ./download_spectra.sh

`download_spectra.sh` will download the observational data for the yet
unfiltered lines of sight to the `data/dr12q/spectra` directory.

Loading and preprocessing spectra
---------------------------------

Now we load these data, continue applying filters, and do some basic
preprocessing. The additional filters are:

* spectra that have no nonmasked pixels in the range [1310, 1325]
  Angstroms (QSO restframe) are filtered, as they cannot be normalized
* spectra with fewer than 400 nonmasked pixels in the range [700,
  5000] Angstroms (QSO restframe) are filtered.

The preprocessing steps are to:

* truncate spectra to only contain pixels in the range [700, 5000]
  Angstroms QSO rest
* normalize flux and noise variance by dividing by the median flux in
  the range [1325, 1390] Angstroms QSO rest

Relevant parameters in `set_parameters` that can be tweaked if
desired:

    % preprocessing parameters
    min_num_pixels = 200;                         % minimum number of non-masked pixels

    % normalization parameters
    normalization_min_lambda = 1325;              % range of rest wavelengths to use   Å
    normalization_max_lambda = 1390;              %   for flux normalization

    % file loading parameters
    loading_min_lambda = 700;                     % range of rest wavelengths to load  Å
    loading_max_lambda = 5000;

When ready, the MATLAB code to preload the spectra is:

    set_parameters;
    release = 'dr12q';

    file_loader = @(plate, mjd, fiber_id) ...
      (read_spec(sprintf('%s/%i/spec-%i-%i-%04i.fits', ...
        spectra_directory(release),                  ...
        plate,                                       ...
        plate,                                       ...
        mjd,                                         ...
        fiber_id)));

    preload_qsos;

The result will be a completed catalog data file,
`data/[release]/processed/catalog.mat`, with complete filtering
information and a file containing preloaded and preprocessed data for
the 162861 nonfiltered spectra,
`data/[release]/processed/preloaded_qsos.mat`.

Building GP models for quasar spectra
-------------------------------------

Now we build our models, including our Gaussian process null model for
quasar emission spectra and our model for spectra containing DLAs.

To build the null model for quasar emission spectra, we need to
indicate a set of spectra to use for training, which should be
nominally DLA-free. Here we select all spectra:

* in DR9
* not removed by our filtering steps during loading
* in the DR9 Lyman-alpha forest catalog, and
* not in the DR9 Lyman-alpha DLA concordance catalog

These particular choices may be accomplished with:

    training_release  = 'dr12q';
    dla_catalog_name = 'dr9q_concordance';
    train_ind = ...
        [' catalog.in_dr9                     & ' ...
         '(catalog.filter_flags == 0)         & ' ...
         ' catalog.los_inds(dla_catalog_name) & ' ...
         '~catalog.dla_inds(dla_catalog_name)'];

After specifying the spectra to use in `training_release` and
`train_ind`, we call `learn_qso_model` to learn the model.
To learn the model, you will need the MATLAB toolbox
[minFunc](https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html)
available from Mark Schmidt.

You should cd to the directory where you installed minFunc to and run:

    addpath(genpath(pwd));
    mexAll;

to initialize minFunc before the first time you use it.

Relevant parameters in `set_parameters` that can be tweaked if
desired:

    % null model parameters
    min_lambda         =     910;                 % range of rest wavelengths to       Å
    max_lambda         =    3000;                 %   model
    dlambda            =    0.25;                 % separation of wavelength grid      Å
    k                  = 20;                      % rank of non-diagonal contribution
    max_noise_variance = 1^2;                     % maximum pixel noise allowed during model training

    % optimization parameters
    initial_c     = 0.1;                          % initial guess for c
    initial_tau_0 = 0.0023;                       % initial guess for τ₀
    initial_beta  = 3.65;                         % initial guess for β
    minFunc_options =               ...           % optimization options for model fitting
        struct('MaxIter',     4000, ...
               'MaxFunEvals', 8000);

When ready, the MATLAB code to learn the null quasar emission model
is:

    training_set_name = 'dr9q_minus_concordance';
    learn_qso_model;

The learned qso model is stored in
`data/[training_release]/processed/learned_qso_model_[training_set_name].mat`.

Processing spectra for continuum fit
------------------------------------

The current fit continuum script only supports user to input thing_ids
into the model.

For example, to load `thing_id = 84523031` quasar,

    % select thing_ids to load into the environment
    selected_thing_ids = [84523031];

    % run the loading script to load only the selected quasars
    load_selected_qsos;

The following variables will be loaded into the environment:

    % Parameters:
    % ----
    % this_rest_wavelengths : λ
    % this_flux             : y
    % this_noise_variance   : v
    % this_omega2           : ω
    % this_mu               : μ
    % this_M                : M, K = MM'

The fitting continuum routine is done in `fitcontinuum.m`. To
run the first quasar in the `selected_thing_ids`, do

    % fit the continuum on the given quasar
    quasar_ind = 1;
    fitcontinuum;

The script will call `conditional_mvnpdf_low_rank` to build a
conditional GP and get:

    % this_continuum (array) : continuum fit with the same range as this_rest_wavelengths
    % Sigma11 (matrix)       : the covariance matrix at Lyman forest region

The description for conditional GP is here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions

To plot the continuum fit, type

    % plot this_continuum with uncertainty (only diagonal terms in the covariance)
    plot_fitcontinuum;

To wrap things up, if I want to plot all of my fits:

    % plot all selected thing_ids
    for quasar_ind = 1:numel(selected_thing_ids)
        fitcontinuum;
        plot_fitcontinuum;
    end

TODO: create another repo for continuum fitting and build a python class to
do it from .fits file directly. The only thing we need is the learning script.
