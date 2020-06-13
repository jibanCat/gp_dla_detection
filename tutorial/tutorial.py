'''
tutorial to plot something using QSOLoaderZ
'''
import numpy as np
from matplotlib import pyplot as plt
from CDDF_analysis.qso_loader import QSOLoaderZ

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)

qsos = QSOLoaderZ(
        preloaded_file="preloaded_zqso_only_qsos.mat",
        catalogue_file="zqso_only_catalog.mat", 
        learned_file="learned_model_outdata_dr9q_minus_concordance_norm_1176-1256.mat",
        processed_file="processed_zqso_only_qsos_dr12q-100_uniformprior.mat",
        dla_concordance="dla_catalog",
        los_concordance="los_catalog",
        sample_file="dla_samples.mat",
        occams_razor=False)


def do_plot_example(qsos, nspec=18):
    '''
    Plot an example spectrum
    '''
    # saving plots: MAP estimate model
    z = qsos.z_map[nspec]

    qsos.plot_this_mu(nspec=nspec,
        num_voigt_lines=3, num_forest_lines=31, z_sample=z)
    plt.ylim(-1, 5)
    save_figure("{}_this_mu_delta_z_{:.2g}".format(
        qsos.thing_ids[nspec], z))

do_plot_example(qsos, nspec=1) # plot the first spectrum

# plot a specific thing_id, 27885089
thing_id_index = np.where( qsos.thing_ids == 27885089 )[0][0]
do_plot_example(qsos, nspec=thing_id_index)

# plot a high zqso example
high_z_index = np.where( qsos.z_qsos > 4.5 )[0]
do_plot_example(qsos, nspec=high_z_index[0])