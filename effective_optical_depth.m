% effective_optical_depth.m : calculate the total effective optical
% from Hydrogen lines

function total_optical_depth = effective_optical_depth(this_wavelengths, ...
    beta, tau_0, z_qso, ...
    all_transition_wavelengths, all_oscillator_strengths, ...
    num_forest_lines)

    lya_wavelength          = all_transition_wavelengths(1);
    lya_oscillator_strength = all_oscillator_strengths(1);

    % To count the effect of Lyman series from higher z,
    % we compute the absorbers' redshifts for all members of the series
    this_lyseries_zs = nan(numel(this_wavelengths), num_forest_lines);

    for l = 1:num_forest_lines
        this_lyseries_zs(:, l) = ...
            (this_wavelengths - all_transition_wavelengths(l)) / ...
            all_transition_wavelengths(l);
    end
    
    % Lyman series absorption effect on the mean-flux
    % apply the lya_absorption after the interpolation because NaN will appear in this_mu
    total_optical_depth = nan(numel(this_wavelengths), num_forest_lines);

    for l = 1:num_forest_lines
        % calculate the oscillator strength for this lyman series member
        this_tau_0 = tau_0 * ...
            all_oscillator_strengths(l)   / lya_oscillator_strength * ...
            all_transition_wavelengths(l) / lya_wavelength;

        total_optical_depth(:, l) = ...
            this_tau_0 .* ( (1 + this_lyseries_zs(:, l)).^beta );

        % indicator function: z absorbers <= z_qso
        % here is different from multi-dla processing script
        % I choose to use zero instead or nan to indicate
        % values outside of the Lyman forest
        indicator = this_lyseries_zs(:, l) <= z_qso;
        total_optical_depth(:, l) = total_optical_depth(:, l) .* indicator;
    end
end
