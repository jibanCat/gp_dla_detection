% redden_mu : reddening mu with the given reddening power factor,
% and re-normalize based on the normalization range

function mu = redden_mu(mu, rest_wavelengths, a, ...
        normalization_min_lambda, normalization_max_lambda)
    mu = mu .* rest_wavelengths^a;

    % select the range to normalize
    ind = (rest_wavelengths >= normalization_min_lambda) & ...
          (rest_wavelengths <= normalization_max_lambda);

    % re-normalize
    this_median = nanmedian(mu(ind));
    mu          = mu / this_median;
end
