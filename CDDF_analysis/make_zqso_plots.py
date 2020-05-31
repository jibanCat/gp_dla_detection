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
        base_directory, processed_directory(release), "processed_zqso_only_qsos_dr12q-100" )
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
