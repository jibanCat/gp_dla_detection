'''
scripts to plot QSOLoaderZ
'''
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from CDDF_analysis.qso_loader import QSOLoaderZ
from CDDF_analysis import make_zqso_plots
from CDDF_analysis.set_parameters import *

matplotlib.use('TkAgg')

delta_z = 0.5

# generate QSOLoader insrance
qsos = make_zqso_plots.generate_qsos()

# z_map versus z_true
index = qsos.plot_z_map(delta_z=delta_z)
make_zqso_plots.save_figure("z_map_vs_z_true_pure-z")
plt.show()

# plot specific example: this example shows -inf posteriors
# at the true zQSO parameter space
nspec = 15

# this plots P(M|D) versus z_samples
qsos.plot_z_sample_posteriors(nspec, dla_samples=False)
plt.show()

# Plot the spectra with this_mu, using MAP z estimate
qsos.plot_this_mu(nspec=nspec, suppressed=True, 
    num_voigt_lines=3, num_forest_lines=6, z_sample=qsos.z_map[nspec])
plt.ylim(-1, 5)
plt.show()

# Plot the spectra with this_mu, using True zQSO
qsos.plot_this_mu(nspec=nspec, suppressed=True, 
    num_voigt_lines=3, num_forest_lines=6, z_sample=qsos.z_qsos[nspec])
plt.ylim(-1, 5)
plt.show()

# for nspec in np.where(index)[0]:
for nspec in range(len(qsos.z_qsos)):
    print("Plotting {}/{} ...".format(nspec, len(qsos.z_qsos)))

    # saving plots: z_samples versus poseteriors
    qsos.plot_z_sample_posteriors(nspec, dla_samples=True)
    plt.savefig("{}_posterior_zqso_samples_delta_z_{}.pdf".format(
            qsos.thing_ids[nspec], delta_z),
            dpi=150, format='pdf')
    # plt.close()
    # plt.clf()
    plt.show()

    # saving plots: MAP estimate model
    qsos.plot_this_mu(nspec=nspec, suppressed=True, 
        num_voigt_lines=3, num_forest_lines=6, z_sample=qsos.z_map[nspec])
    plt.ylim(-1, 5)        
    make_zqso_plots.save_figure(
        "{}_this_mu_delta_z_{}_ZMAP".format(
            qsos.thing_ids[nspec], delta_z))
    # plt.close()
    # plt.clf()
    plt.show()

    # saving plots: True QSO rest-frame
    qsos.plot_this_mu(nspec=nspec, suppressed=True, 
        num_voigt_lines=3, num_forest_lines=6, z_sample=qsos.z_qsos[nspec])
    plt.ylim(-1, 5)    
    make_zqso_plots.save_figure(
        "{}_this_mu_delta_z_{}_ZTrue".format(
            qsos.thing_ids[nspec], delta_z))
    # plt.close()
    # plt.clf()
    plt.show()

# inspect the this_wavelength due to the normalisation is weird
this_wavelengths    = qsos.find_this_wavelengths(nspec)
this_noise_variance = qsos.find_this_noise_variance(nspec)
this_flux           = qsos.find_this_flux(nspec)

this_rest_wavelengths = this_wavelengths / ( 1 + qsos.z_qsos[nspec] )

ind = ( (this_rest_wavelengths >= qsos.normalization_min_lambda) & 
    (this_rest_wavelengths <= qsos.normalization_max_lambda))

this_median    = np.nanmedian(this_flux[ind])

# new prior range
# ( min(this_wavelength)/lya_wavelength - 1, max(this_wavelength)/lya_wavelength - 1)

# Note:
# ----
# 1) the best solution is to avoid sampling zQSO too large or too small
#    However, since zQSO is unknown, so we never know this z_sample
#    is too large or too small.
# 2) second solution is to weaken the GP likelihoods for lambda > 1216
#    Lots of examples fit the full model to lambda > 1216; the major reason
#    is metal region has smaller omega and most flux are flat, so the resultant
#    likelihood is always larger than considering lambda < 1216.
#    The way to weaken could be tuning the interpolation. We actually want more
#    flux points from Lya regions since we want our GP model across Lya and metal
#    regions. Instead of moving lambda_obs to lambda_rest, we can try to do
#    model_lambda_rest to lambda_obs. Stick with delta_lambda_obs = 0.25, 
#    In this case we have more points at Lya region.
# 3) third idea is to make the null model stronger at Lya region: add subDLA,
#    and replace or weaken null model. Actually, adding subDLA should help to
#    constrain the Lya region, which is currently poorly constrain by the null
#    model.
#  
#  I still fill (1) - finding new zQSO prior range per spectrum - is the most 
#  robust way though not quite Bayesian. And it is hard to tell if z_sample too
#  large or too small

