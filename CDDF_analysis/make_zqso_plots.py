'''
Make plots for Z estimate paper
'''
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from astropy.io import fits

from .set_parameters import *
from .qso_loader import QSOLoaderZ

# change fontsize
matplotlib.rcParams.update({'font.size' : 14})

# matplotlib.use('PDF')

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)

def generate_qsos(base_directory="", release="dr12q",
        dla_concordance="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        los_concordance="data/dla_catalogs/dr9q_concordance/processed/los_catalog",
        suppressed=False):
    '''
    Return a QSOLoader instances : zqsos
    '''
    preloaded_file = os.path.join( 
        base_directory, processed_directory(release), "preloaded_zqso_only_qsos.mat")
    processed_file  = os.path.join(
        base_directory, processed_directory(release), "processed_zqso_only_qsos_dr12q-100_uniformprior.mat" )
    catalogue_file = os.path.join(
        base_directory, processed_directory(release), "zqso_only_catalog.mat")
    learned_file   = os.path.join(
        base_directory, processed_directory(release), "learned_zqso_only_model_outdata_full_dr9q_minus_concordance_norm_1176-1256.mat")
    sample_file    = os.path.join(
        base_directory, processed_directory(release), "dla_samples.mat")

    qsos_zqsos = QSOLoaderZ(
        preloaded_file=preloaded_file, catalogue_file=catalogue_file, 
        learned_file=learned_file, processed_file=processed_file,
        dla_concordance=dla_concordance, los_concordance=los_concordance,
        sample_file=sample_file, occams_razor=False, suppressed=suppressed)

    return qsos_zqsos

def do_procedure_plots(qsos, model_min_lambda=910, model_max_lambda=3000):
    # scaling factor between rest_wavelengths to pixels
    min_lambda = model_min_lambda - 10
    max_lambda = model_max_lambda + 10
    scale = 1 / ( max_lambda - min_lambda )

    # compare different learned mus
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.plot(
        qsos.GP.rest_wavelengths, qsos.GP.mu, label=r"$\mu$ (prior mean)")

    ax.legend()
    ax.set_xlabel(r"Restframe wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
    ax.set_ylabel(r"Normalized flux")
    ax.set_xlim([min_lambda, max_lambda])

    ax02 = ax.twiny()
    ax02.set_xticks(
        [
            (lyman_limit     - min_lambda) * scale,
            (lyb_wavelength  - min_lambda) * scale,
            (lya_wavelength  - min_lambda) * scale,
            (civ_wavelength  - min_lambda) * scale,
            (siiv_wavelength - min_lambda) * scale,
            (ciii_wavelength - min_lambda) * scale,
            (mgii_wavelength - min_lambda) * scale,
        ]
    )
    ax02.set_xticklabels([r"Ly $\infty$", r"Ly $\beta$", r"Ly $\alpha$", 
                          r"C$_{IV}$", r"Si$_{IV}$", r"C$_{III}$",
                          r"Mg$_{II}$"])
    plt.tight_layout()
    save_figure("GP_mu")
    plt.clf()

    # plotting covariance matrix
    min_lambda = model_min_lambda 
    max_lambda = model_max_lambda
    scale = np.shape(qsos.GP.C)[0] / ( max_lambda - min_lambda )

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(qsos.GP.C, origin="lower", cmap="gray_r")
    ax.set_xticks(
        [
         (lyman_limit    - min_lambda) * scale,
         (lyb_wavelength - min_lambda) * scale,
         (lya_wavelength - min_lambda) * scale,
         (civ_wavelength  - min_lambda) * scale,
         (siiv_wavelength - min_lambda) * scale,
         (ciii_wavelength - min_lambda) * scale,
         (mgii_wavelength - min_lambda) * scale,
        ]
    )
    ax.set_xticklabels([r"Ly$\infty$", r"Ly$\beta$", r"Ly$\alpha$", 
                        r"C$_{IV}$", r"Si$_{IV}$", r"C$_{III}$",
                        r"Mg$_{II}$"])
    ax.set_yticks(
        [
         (lyman_limit    - min_lambda) * scale,
         (lyb_wavelength - min_lambda) * scale,
         (lya_wavelength - min_lambda) * scale,
         (civ_wavelength  - min_lambda) * scale,
         (siiv_wavelength - min_lambda) * scale,
         (ciii_wavelength - min_lambda) * scale,
         (mgii_wavelength - min_lambda) * scale,
        ]
    )
    ax.set_yticklabels([r"Ly$\infty$", r"Ly$\beta$", r"Ly$\alpha$", 
                        r"C$_{IV}$", r"Si$_{IV}$", r"C$_{III}$",
                        r"Mg$_{II}$"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)    
    plt.tight_layout()
    save_figure("covariance_matrix")
    plt.clf()

def do_plot_example(qsos, nspec=18):
    '''
    Plot an example spectrum, better to have a good lambda coverage
    '''
    # saving plots: MAP estimate model
    z = qsos.z_map[nspec]

    qsos.plot_this_mu(nspec=nspec,
        num_voigt_lines=3, num_forest_lines=0, z_sample=z,
        suppressed=qsos.suppressed)
    plt.ylim(-1, 5)
    save_figure("{}_this_mu_delta_z_{:.2g}".format(
        qsos.thing_ids[nspec], z))
    
    # a plot with a wrong zsample
    z = 3.5

    qsos.plot_this_mu(nspec=nspec,
        num_voigt_lines=3, num_forest_lines=0, z_sample=z,
        suppressed=qsos.suppressed)
    plt.ylim(-1, 5)
    save_figure("{}_this_mu_delta_z_{:.2g}".format(
        qsos.thing_ids[nspec], z))
    

def do_velocity_dispersions(qsos, dr12q_fits='data/dr12q/distfiles/DR12Q.fits'):
    '''
    Reproduce the figure 7 in SDSS DR12Q paper, with Z_MAP
    '''
    dr12q = fits.open(dr12q_fits)
    
    # acquire the table data in SDSS DR12Q paper; Table 4.
    table = dr12q[1].data

    Z_VI   = table['Z_VI']
    Z_PIPE = table['Z_PIPE']
    Z_PCA  = table['Z_PCA']
    Z_CIV  = table['Z_CIV']
    Z_CIII  = table['Z_CIII']
    Z_MGII = table['Z_MGII']

    # filter out non-detections (were labeled as -1)
    ind = [ Z != -1 for Z in (Z_VI, Z_PIPE, Z_PCA, Z_CIV, Z_CIII, Z_MGII) ]
    ind = np.all(ind, axis=0)

    z_map_ind = ind[qsos.test_ind]

    # include the test_ind we applied during testing
    ind = ind & qsos.test_ind

    Z_VI   = Z_VI[ind]
    Z_PIPE = Z_PIPE[ind]
    Z_PCA  = Z_PCA[ind]
    Z_CIV  = Z_CIV[ind]
    Z_CIII  = Z_CIII[ind]
    Z_MGII = Z_MGII[ind]
    
    z_map = qsos.z_map[z_map_ind]

    bins = np.linspace(-7500, 7500, 15000 // 100)

    plt.hist( z_to_kms( Z_VI - Z_PCA ), bins=bins, histtype='step', label='Z_VI')
    plt.hist( z_to_kms( Z_MGII - Z_PCA ), bins=bins, histtype='step', label='Z_MGII')
    plt.hist( z_to_kms( Z_PIPE - Z_PCA ), bins=bins, histtype='step', label='Z_PIPE')
    plt.hist( z_to_kms( Z_CIV - Z_PCA ), bins=bins, histtype='step', label='Z_CIV')
    plt.hist( z_to_kms( Z_CIII - Z_PCA ), bins=bins, histtype='step', label='Z_CIII')
    plt.xlabel('$\Delta v (z_x - z_{PCA})$ (km/s)')
    plt.ylabel('Number of quasars')
    plt.legend()
    plt.tight_layout()
    save_figure("SDSS_DR12Q_Figure7")
    plt.clf()
    plt.close()

    plt.hist( z_to_kms( Z_VI - Z_PCA ), bins=bins, histtype='step', label='Z_VI', ls='--')
    plt.hist( z_to_kms( Z_MGII - Z_PCA ), bins=bins, histtype='step', label='Z_MGII', ls='--')
    plt.hist( z_to_kms( Z_PIPE - Z_PCA ), bins=bins, histtype='step', label='Z_PIPE', ls='--')
    plt.hist( z_to_kms( z_map - Z_PCA ), bins=bins, histtype='step', label='$z_{MAP}$', lw=2)
    plt.xlabel('$\Delta v (z_x - z_{PCA})$ (km/s)')
    plt.ylabel('Number of quasars')
    plt.legend()
    plt.tight_layout()
    save_figure("SDSS_DR12Q_Figure7_w_ZMAP")

    print("{} QSOs in total".format(len(z_map)))
    print('Median: z_to_kms( Z_VI - Z_PCA )',   np.median(z_to_kms( Z_VI - Z_PCA )))
    print('Median: z_to_kms( Z_MGII - Z_PCA )', np.median(z_to_kms( Z_MGII - Z_PCA )))
    print('Median: z_to_kms( Z_PIPE - Z_PCA )', np.median(z_to_kms( Z_PIPE - Z_PCA )))
    print('Median: z_to_kms( z_map - Z_PCA )',  np.median(z_to_kms( z_map - Z_PCA )))
