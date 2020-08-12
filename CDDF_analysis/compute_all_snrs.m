% compute_all_snrs.m : a MATLAB version of compute_all_snrs function in calc_cddf.py
%   but instead of loading file everytime, load all the file into memory.
%   It's weird to place this file here, but for users who experience hard time
%   computing .calc_cddf.compute_all_snrs, this function can run faster.
%   TODO: add to the multi-DLA MATLAB code in the future.
%
%   to run:
%   >> # in MATLAB, root folder of the repo
%   >> addpath multi_dlas/
%   >> set_parameters_multi;
%   >> % prior setting
%   >> release          = 'dr14q';
%   >> trainin_set_name = 'multi_meanfluxdr14q';

% load processed file variables
variables_to_load = {'test_ind', 'min_z_dlas', 'max_z_dlas'};

load(sprintf('%s/processed_qsos_%s',      ...
            processed_directory(release), ...
            trainin_set_name),            ...
    variables_to_load{:});

% load preprocessed file
variables_to_load = {'all_wavelengths', 'all_flux', 'all_normalizers', 'all_noise_variance'};

load(sprintf('%s/preloaded_qsos.mat',      ...
            processed_directory(release)), ...
    variables_to_load{:});

% construct snrs array with the same length of processed_qsos
num_qsos = sum(test_ind);
snrs     = nan(num_qsos, 1);
real_index = find(test_ind ~= 0);

parfor nspec = 1:num_qsos
    nspec_real = real_index(nspec, 1);

    zmin = min_z_dlas(nspec);
    zmax = max_z_dlas(nspec);

    this_wavelengths    = all_wavelengths{nspec_real};
    this_flux           = all_flux{nspec_real};
    this_normalizer     = all_normalizers(nspec_real);
    this_noise_variance = all_noise_variance{nspec_real};

    % took the convention of sbird's calc_cddf.find_snr
    ipix = (this_wavelengths >  lya_wavelength * (1 + zmax));

    this_flux           = this_flux(ipix);
    this_noise_variance = this_noise_variance(ipix);

    norm_inds = (this_flux / this_normalizer) < 0.1 * this_normalizer;

    if sum(norm_inds) > 0
        this_flux(norm_inds) = 0.1 * this_normalizer;
    else
        norm_inds = (this_flux < 0.1);
        this_flux(norm_inds) = 0.1;
    end

    snrs(nspec, 1) = 1 / nanmedian( sqrt(this_noise_variance) ./ abs(this_flux) );
end

variables_to_save = {'snrs'};

filename = sprintf('%s/snrs_qsos_%s.mat',       ...
                  processed_directory(release), ...
                  trainin_set_name);

save(filename, variables_to_save{:}, '-v7.3');
