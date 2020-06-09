'''
Make plots for Z estimate paper
'''
import os
import numpy as np
from scipy.stats import pearsonr
import matplotlib
from matplotlib import pyplot as plt
from .set_parameters import *
from .qso_loader_zqsos2 import QSOLoaderZDLAs
from .dla_data import dla_data

# change fontsize
matplotlib.rcParams.update({'font.size' : 14})

# matplotlib.use('PDF')

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)

def generate_qsos(base_directory="", release="dr12q",
        processed_filename="processed_qsos_zqsos_sbird_dr12q-100_norm_1176-1256.mat",
        learned_filename="learned_model_outdata_dr9q_minus_concordance_norm_1176-1256.mat",
        dla_concordance="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        los_concordance="data/dla_catalogs/dr9q_concordance/processed/los_catalog"):
    '''
    Return a QSOLoader instances : zqsos
    '''
    preloaded_file = os.path.join( 
        base_directory, processed_directory(release), "preloaded_zqso_only_qsos.mat")
    processed_file  = os.path.join(
        base_directory, processed_directory(release), processed_filename )
    catalogue_file = os.path.join(
        base_directory, processed_directory(release), "zqso_only_catalog.mat")
    learned_file   = os.path.join(
        base_directory, processed_directory(release), learned_filename)
    sample_file    = os.path.join(
        base_directory, processed_directory(release), "dla_samples.mat")

    qsos_zqsos = QSOLoaderZDLAs(
        preloaded_file=preloaded_file, catalogue_file=catalogue_file, 
        learned_file=learned_file, processed_file=processed_file,
        dla_concordance=dla_concordance, los_concordance=los_concordance,
        sample_file=sample_file, occams_razor=False)

    return qsos_zqsos

def do_ROC(qsos, occams_razor=1):
    '''
    Plot Two ROC curves to demonstrate the difference

    Parameters:
    ----
    occams_razor : N > 0, only for multi-DLA
    '''
    TPR, FPR   = qsos.make_ROC(qsos.dla_catalog, occams_razor=occams_razor)
    
    from scipy.integrate import cumtrapz

    AUC  = - cumtrapz(TPR, x=FPR)[-1]

    plt.plot(FPR,  TPR,  color="C1", label="current;  AUC: {:.3g}".format(AUC))
    plt.xlabel("False positive rate (FPR)")
    plt.ylabel("True positive rate (TPR)")
    plt.legend()
    save_figure("ROC_z")
    plt.clf()

def do_MAP_hist2d(qsos):
    '''
    Do the hist2d in between z_true vs z_map
    '''
    map_z_dlas, true_z_dlas, map_log_nhis, true_log_nhis, real_index = qsos.make_MAP_hist2d(
        p_thresh=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))
    (h1, x1edges, y1edges, im1) = ax1.hist2d(map_z_dlas, true_z_dlas,
        bins = int(np.sqrt(map_z_dlas.shape[0])), cmap='viridis')
    # a perfect prediction straight line
    z_dlas_plot = np.linspace(2.0, 5.0, 100)
    ax1.plot(z_dlas_plot, z_dlas_plot)
    ax1.set_xlabel(r"$z_{{DLA,MAP}}$")
    ax1.set_ylabel(r"$z_{{DLA,concordance}}$")
    fig.colorbar(im1, ax=ax1)

    (h2, x2edges, y2edges, im2) = ax2.hist2d(map_log_nhis, true_log_nhis,
        bins = int(np.sqrt(map_z_dlas.shape[0])), cmap='viridis')

    # a perfect prediction straight line
    log_nhi_plot = np.linspace(20, 22.5, 100)
    ax2.plot(log_nhi_plot, log_nhi_plot)

    # # 3rd polynomial fit
    # poly_fit =  np.poly1d( np.polyfit(map_log_nhis, true_log_nhis, 4 ) )
    # ax2.plot(log_nhi_plot, poly_fit(log_nhi_plot), color="white", ls='--')

    ax2.set_xlabel(r"$\log N_{{HI,MAP}}$")
    ax2.set_ylabel(r"$\log N_{{HI,concordance}}$")
    ax2.set_xlim(20, 22.5)
    ax2.set_ylim(20, 22.5)
    fig.colorbar(im2, ax=ax2)
    save_figure("MAP_hist2d_concordance")

    print("Pearson Correlation for (map_z_dlas,   true_z_dlas) : ",
        pearsonr(map_z_dlas, true_z_dlas))
    print("Pearson Correlation for (map_log_nhis, true_log_nhis) : ",
        pearsonr(map_log_nhis, true_log_nhis))

    # examine the pearson correlation per log nhi bins
    log_nhi_bins = [20, 20.5, 21, 23]

    for (min_log_nhi, max_log_nhi) in zip(log_nhi_bins[:-1], log_nhi_bins[1:]):
        ind  =  (map_log_nhis > min_log_nhi) & (map_log_nhis < max_log_nhi)
        ind = ind & (true_log_nhis > min_log_nhi) & (true_log_nhis < max_log_nhi)
        
        print("Map logNHI Bin [{}, {}] Pearson Correlation for (map_log_nhis, true_log_nhi) : ".format(
            min_log_nhi, max_log_nhi),
            pearsonr(map_log_nhis[ind], true_log_nhis[ind]))



def do_MAP_CDDF(qsos, subdir, zmax=5, snr_thresh=-1):
    '''
    Plot the column density distribution function of Z Esitmation DR12
    '''
    dla_data.noterdaeme_12_data()
    (l_N, cddf) = qsos.plot_cddf(
        zmax=zmax, snr_thresh=snr_thresh, color="blue")
    np.savetxt(
        os.path.join(subdir, "cddf_zestimation_all.txt"),
        (l_N, cddf))
    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_zestimation"))
    plt.clf()

    # Evolution with redshift
    (l_N, cddf) = qsos.plot_cddf(
        zmin=4, zmax=5, label="4-5", color="brown")
    np.savetxt(
        os.path.join(subdir, "cddf_zestimation_z45.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf(
        zmin=3, zmax=4, label="3-4", color="black")
    np.savetxt(
        os.path.join(subdir, "cddf_zestimation_z34.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf(
        zmin=2.5, zmax=3, label="2.5-3", color="green")
    np.savetxt(
        os.path.join(subdir, "cddf_zestimation_z253.txt"), (l_N, cddf))
    (l_N, cddf) = qsos.plot_cddf(
        zmin=2, zmax=2.5, label="2-2.5", color="blue")
    np.savetxt(
        os.path.join(subdir, "cddf_zestimation_z225.txt"), (l_N, cddf))

    plt.xlim(1e20, 1e23)
    plt.ylim(1e-28, 5e-21)
    plt.legend(loc=0)
    save_figure(os.path.join(subdir, "cddf_zz_zestimation"))
    plt.clf()    

def do_MAP_dNdX(qsos, subdir, zmax=5, snr_thresh=-1):
    '''
    Plot line density of Z Estimation DR12
    '''
    dla_data.dndx_not()
    dla_data.dndx_pro()
    z_cent, dNdX = qsos.plot_line_density(
        zmax=zmax, snr_thresh=snr_thresh)
    np.savetxt(os.path.join(subdir, "dndx_zestimation_all.txt"), (z_cent, dNdX))

    plt.legend(loc=0)
    plt.ylim(0, 0.16)
    save_figure(os.path.join(subdir, "dndx_zestimation"))
    plt.clf()
