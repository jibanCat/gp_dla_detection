'''
Reproduce the plots in Leah et al (2020) paper
'''
from CDDF_analysis import make_zqso_plots

# make sure every loaded files are in the correct directories
# otherwise, using the loading method in tutorial.py
qsos = make_zqso_plots.generate_qsos()

# plot GP mu and kernel
make_zqso_plots.do_procedure_plots(qsos)

# plot figure 6 & 7 in the paper
make_zqso_plots.do_plot_thing_ids(qsos, selected_thing_ids=[544031279, 27885089])

# reproduce z_map - z_PCA plot
make_zqso_plots.do_velocity_dispersions(qsos, dr12q_fits='data/dr12q/distfiles/DR12Q.fits')
