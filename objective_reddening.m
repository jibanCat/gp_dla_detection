% objective_reddening: the reddening objective function to find the power-law index
% of the reddening of the qso spectrum:
% 
% mu' := mu .* rest_lambda^alpha

function loss = objective_reddening(rest_wavelengths, this_rest_fluxes, ...
        this_lya_absorptions, mu, a, ...
        normalization_min_lambda, normalization_max_lambda)
    % select the range to normalize
    ind = (rest_wavelengths >= normalization_min_lambda) & ...
          (rest_wavelengths <= normalization_max_lambda);

    this_median = nanmedian(this_rest_fluxes(ind));
    this_rest_fluxes = this_rest_fluxes / this_median;

    % apply reddening to the mu and re-normalize
    mu = mu .* rest_wavelengths^a;

    this_median = nanmedian(mu(ind));
    mu = mu / this_median;

    % apply effective optical depth
    mu = mu .* this_lya_absorptions;

    % just chi^2 since no covariance information
    loss = nansum( (this_rest_fluxes - mu)^2 );
end
