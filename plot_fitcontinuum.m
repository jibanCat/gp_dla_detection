% plot_fitcontinuum.m : plot the fitted continuum with diagnoal uncertainty
% build the uncertainty, only diag elements

sigma1 = sqrt(diag(Sigma11));
assert(numel(sigma1) == numel(d1))

sigma2 = sqrt(diag( this_M2 * this_M2' ) + d2);
assert(numel(sigma2) == numel(y2))

this_sigma = cat(1, sigma1, sigma2);

% prepare fill_between plot
this_mu_plus_sigma  = this_continuum + this_sigma;
this_mu_minus_sigma = this_continuum - this_sigma;
this_rest_wavelengths_plot = [this_rest_wavelengths', fliplr(this_rest_wavelengths')];
fill_between = [this_mu_minus_sigma', fliplr(this_mu_plus_sigma')];

figure('Renderer', 'painters', 'Position', [10 10 2000 900])
hold on;
    plot(this_rest_wavelengths, this_flux, 'Color', '#0072BD');
    plot(this_rest_wavelengths, this_continuum, 'Color', '#D95319', 'LineWidth', 2);
    plot(this_rest_wavelengths, this_mu, 'Color', '#EDB120', 'LineWidth', 2);
    f = fill(this_rest_wavelengths_plot, fill_between, 'red', 'EdgeColor','none');
    set(f,'FaceAlpha',0.2);
    ylim([-1 5]);

    % label annotations
    xlabel('$\lambda_{rest}$', 'FontSize', 20, 'Interpreter','latex');
    ylabel('normalised flux',  'FontSize', 20, 'Interpreter','latex');

    legend('observed flux', 'conditional GP', 'GP prior', ...
        'Interpreter','latex', 'FontSize', 16);
    title(sprintf('thingID = %d, zQSO = %.2f', selected_thing_ids(quasar_ind), z_qso), ...
        'FontSize', 20, 'Interpreter','latex'); 
    saveas(gcf, sprintf('Continumm_fit_%d.svg', selected_thing_ids(quasar_ind)) )
    set(gcf, 'Visible', 'off')
hold off;

figure('Renderer', 'painters', 'Position', [10 10 2000 900])
hold on;
    plot(this_rest_wavelengths, this_continuum, 'Color', '#D95319', 'LineWidth', 2); 
    plot(this_rest_wavelengths, this_mu_mf, 'Color', '#4DBEEE', ...
        'LineStyle', '--', 'LineWidth', 2); 
    plot(this_rest_wavelengths, this_mu, 'Color', '#4DBEEE', 'LineWidth', 2); 
    ylim([-1 5]);

    % label annotations
    xlabel('$\lambda_{rest}$', 'FontSize', 20, 'Interpreter','latex');
    ylabel('normalised flux',  'FontSize', 20, 'Interpreter','latex');

    legend('conditional GP', 'GP prior', 'GP prior MF suppressed', ...
        'Interpreter','latex', 'FontSize', 16);
    title(sprintf('thingID = %d, zQSO = %.2f', selected_thing_ids(quasar_ind), z_qso), ...
        'FontSize', 20, 'Interpreter','latex');
    saveas(gcf, sprintf('GP_priors_%d.svg', selected_thing_ids(quasar_ind)) )
    set(gcf, 'Visible', 'off')
    set(gcf, 'Visible', 'off')
hold off;
close all
