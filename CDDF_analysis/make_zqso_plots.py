'''
Make plots for Z estimate paper
'''
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from .set_parameters import *
from .qso_loader_zqsos2 import QSOLoaderZDLAs

# change fontsize
matplotlib.rcParams.update({'font.size' : 14})

# matplotlib.use('PDF')

save_figure = lambda filename : plt.savefig("{}.pdf".format(filename), format="pdf", dpi=300)

def generate_qsos(base_directory="", release="dr12q",
        processed_filename="processed_qsos_zqsos2_sbird_dr12q-100.mat",
        learned_filename="learned_model_outdata_dr9q_minus_concordance_norm_1176-1256.mat",
        dla_concordance="data/dla_catalogs/dr9q_concordance/processed/dla_catalog",
        los_concordance="data/dla_catalogs/dr9q_concordance/processed/los_catalog"):
    '''
    Return a QSOLoader instances : zqsos
    '''
    preloaded_file = os.path.join( 
        base_directory, processed_directory(release), "preloaded_qsos.mat")
    processed_file  = os.path.join(
        base_directory, processed_directory(release), processed_filename )
    catalogue_file = os.path.join(
        base_directory, processed_directory(release), "catalog.mat")
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
