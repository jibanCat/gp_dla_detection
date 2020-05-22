import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from scipy.interpolate import interp1d
import h5py

from .set_parameters import *
from .qso_loader import QSOLoader, GPLoader, make_fig

class QSOLoaderZDLAs(QSOLoader):
    '''
    A specific QSOLoader for Z estimation + DLA statistics
    '''
    # include the normaliser since it is not in the processed file
    normalization_min_lambda = 1325                 # range of rest wavelengths to use   Å
    normalization_max_lambda = 1390                 #   for flux normalization


    def __init__(self, preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat", 
            learned_file="learned_qso_model_dr9q_minus_concordance.mat", processed_file="processed_qsos_dr12q.mat",
            dla_concordance="dla_catalog", los_concordance="los_catalog", sample_file="dla_samples.mat",
            occams_razor=False, small_file = True):
        self.preloaded_file = h5py.File(preloaded_file, 'r')
        self.catalogue_file = h5py.File(catalogue_file, 'r')
        self.learned_file   = h5py.File(learned_file,   'r')
        self.processed_file = h5py.File(processed_file, 'r')
        self.sample_file    = h5py.File(sample_file, 'r')

        self.occams_razor = occams_razor

        # load processed data
        self.model_posteriors = self.processed_file['model_posteriors'][()].T

        self.p_dlas           = self.processed_file['p_dlas'][0, :]
        self.p_no_dlas        = self.processed_file['p_no_dlas'][0, :]

        # test_set prior inds : organise arrays into the same order using selected test_inds
        self.test_ind = self.processed_file['test_ind'][0, :].astype(np.bool) #size: (num_qsos, )
        self.test_real_index = np.nonzero( self.test_ind )[0]

        if small_file:
            # note the assumption here is we have samples from 1 ~ num_quasars
            num_quasars = len(self.p_dlas)
            self.test_real_index = self.test_real_index[:num_quasars]

            # filter out test_ind which not in the range of testing
            ind = np.arange(len(self.test_ind)) > max(self.test_real_index)
            self.test_ind[ind] = False

        # z code specific vars
        self.z_map     = self.processed_file['z_map'][0, :]
        self.z_true    = self.processed_file['z_true'][0, :]
        self.dla_true  = self.processed_file['dla_true'][0, :]
        self.z_dla_map = self.processed_file['z_dla_map'][0, :]
        self.n_hi_map  = self.processed_file['n_hi_map'][0, :]
        self.snrs      = self.processed_file['signal_to_noise'][0, :]
        
        # memory free loading; using disk I/O to load sample posteriors
        self.sample_log_posteriors_no_dla = self.processed_file[
            'sample_log_posteriors_no_dla']
        self.sample_log_posteriors_dla    = self.processed_file[
            'sample_log_posteriors_dla']

        # store thing_ids based on test_set prior inds
        self.thing_ids = self.catalogue_file['thing_ids'][0, :].astype(np.int)
        self.thing_ids = self.thing_ids[self.test_ind]

        # plates, mjds, fiber_ids
        self.plates    = self.catalogue_file['plates'][0, :].astype(np.int)
        self.mjds      = self.catalogue_file['mjds'][0, :].astype(np.int)
        self.fiber_ids = self.catalogue_file['fiber_ids'][0, :].astype(np.int)

        self.plates    = self.plates[self.test_ind]
        self.mjds      = self.mjds[self.test_ind]
        self.fiber_ids = self.fiber_ids[self.test_ind]

        # store small arrays
        self.z_qsos     = self.catalogue_file['z_qsos'][0, :]
        self.snrs_cat   = self.catalogue_file['snrs'][0, :]

        self.z_qsos = self.z_qsos[self.test_ind]
        self.snrs_cat   = self.snrs_cat[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        if occams_razor != False:
            self.model_posteriors = self._occams_model_posteriors(self.model_posteriors, self.occams_razor)
            self.p_dlas    = self.model_posteriors[:, 1:].sum(axis=1)
            self.p_no_dlas = self.model_posteriors[:, :1].sum(axis=1)

        # build a MAP number of DLAs array
        # construct a reference array of model_posteriors in Roman's catalogue for computing ROC curve
        multi_p_dlas    = self.model_posteriors # shape : (num_qsos, 2 + num_dlas)

        dla_map_model_index = np.argmax( multi_p_dlas, axis=1 )
        multi_p_dlas = multi_p_dlas[ np.arange(multi_p_dlas.shape[0]), dla_map_model_index ]

        # remove all NaN slices from our sample
        nan_inds = np.isnan( multi_p_dlas )

        self.test_ind[self.test_ind == True] = ~nan_inds

        multi_p_dlas          = multi_p_dlas[~nan_inds]
        dla_map_model_index   = dla_map_model_index[~nan_inds]
        self.test_real_index  = self.test_real_index[~nan_inds]
        self.model_posteriors = self.model_posteriors[~nan_inds, :]
        self.p_dlas           = self.p_dlas[~nan_inds]
        self.p_no_dlas        = self.p_no_dlas[~nan_inds]
        self.thing_ids        = self.thing_ids[~nan_inds]
        self.plates           = self.plates[~nan_inds]
        self.mjds             = self.mjds[~nan_inds]
        self.fiber_ids        = self.fiber_ids[~nan_inds]
        self.z_qsos           = self.z_qsos[~nan_inds]
        self.snrs_cat         = self.snrs_cat[~nan_inds]
        self.snrs             = self.snrs[~nan_inds]

        self.z_map     = self.z_map[~nan_inds]
        self.z_true    = self.z_true[~nan_inds]
        self.dla_true  = self.dla_true[~nan_inds]
        self.z_dla_map = self.z_dla_map[~nan_inds]
        self.n_hi_map  = self.n_hi_map[~nan_inds]

        self.nan_inds = nan_inds
        assert np.any( np.isnan( multi_p_dlas )) == False

        self.multi_p_dlas        = multi_p_dlas
        self.dla_map_model_index = dla_map_model_index

        # store learned GP models
        self.GP = GPLoader(
            self.learned_file['rest_wavelengths'][:, 0],
            self.learned_file['mu'][:, 0],
            self.learned_file['M'][()].T,
            self.learned_file['log_tau_0'][0, 0],
            self.learned_file['log_beta'][0, 0],
            self.learned_file['log_c_0'][0, 0],
            self.learned_file['log_omega'][:, 0]
        )

        # load dla_catalog
        self.load_dla_concordance(dla_concordance, los_concordance)        

        # make sure everything sums to unity
        assert np.all( 
            (self.model_posteriors.sum(axis=1) < 1.2) * 
            (self.model_posteriors.sum(axis=1) > 0.8) )

        # loading DLA samples for plotting z_true comparing to marginalised likelihoods
        self.offset_samples_qso = self.sample_file['offset_samples_qso'][:, 0]
        self.log_nhi_samples    = self.sample_file['log_nhi_samples'][:, 0]
        self.offset_samples     = self.sample_file['offset_samples'][:, 0]

    @staticmethod
    def find_large_delta_z(z_map, z_true, delta_z):
        '''
        return the ind with large delta_z between MAP estimate and z_true
        '''
        ind = np.abs( z_map - z_true ) > delta_z
        return ind

    def plot_z_map(self, zmin=2.15, zmax=6, delta_z=1):
        '''
        plot the z_map as x-axis and z_true as y-axis
        '''
        plt.scatter(self.z_map, self.z_true)

        ind = self.find_large_delta_z(self.z_map, self.z_true, delta_z)
        
        plt.scatter(self.z_map[ind], self.z_true[ind],
            color="red",
            label="z_delta > {:.2g}".format(delta_z))
        print("miss z estimate : {:.2g}%".format(ind.sum() / ind.shape[0] * 100))
        print("index with larger than delta_z:", np.where(~self.nan_inds)[0][ind] )

        plt.plot(np.linspace(zmin, zmax, 100), np.linspace(zmin, zmax, 100),
            color='C1', ls='--')
        plt.xlim(zmin, zmax)
        plt.ylim(zmin, zmax)
        
        plt.xlabel(r"$z_\mathrm{MAP}$")
        plt.ylabel(r"$z_\mathrm{true}$")

        return ind

    def plot_z_sample_posteriors(self, nspec, dla_samples=False):
        '''
        plot the z_samples versus sample posteriors
        '''
        # loading from files to save memory
        nspec_nan = np.where(~self.nan_inds)[0][nspec]
        this_sample_log_posteriors_no_dla = self.sample_log_posteriors_no_dla[:, nspec_nan]
        this_sample_log_posteriors_dla    = self.sample_log_posteriors_dla[:, nspec_nan]

        assert self.processed_file['z_true'][0, nspec_nan] == self.z_true[nspec]

        this_sample_log_posteriors = logsumexp([
            this_sample_log_posteriors_no_dla, this_sample_log_posteriors_dla], axis=0)

        make_fig()

        plt.scatter(self.offset_samples_qso,
            this_sample_log_posteriors_no_dla,
            color="black", label="P(¬DLA | D)", 
            rasterized=True)
        
        if dla_samples:
            plt.scatter(self.offset_samples_qso,
                this_sample_log_posteriors,
                color="red", label="P(DLA + ¬DLA | D)", alpha=0.5,
                rasterized=True)

        # plot verticle lines corresponding to metal lines miss fitted lya
        z_ovi  = ovi_wavelength  * (1 + self.z_true[nspec]) / lya_wavelength - 1
        z_lyb  = lyb_wavelength  * (1 + self.z_true[nspec]) / lya_wavelength - 1
        z_oi   = oi_wavelength   * (1 + self.z_true[nspec]) / lya_wavelength - 1
        z_siiv = siiv_wavelength * (1 + self.z_true[nspec]) / lya_wavelength - 1
        z_civ  = civ_wavelength  * (1 + self.z_true[nspec]) / lya_wavelength - 1

        non_inf_ind = ~np.isinf( this_sample_log_posteriors_no_dla )

        ymin = this_sample_log_posteriors_no_dla[non_inf_ind].min()
        ymax = this_sample_log_posteriors_no_dla[non_inf_ind].max()

        plt.vlines([self.z_true[nspec], z_ovi, z_lyb, z_oi, z_siiv, z_civ],
            ymin, ymax, color="red", ls='--')

        plt.text(z_ovi,  ymax, r"Z_OVI",  rotation=90, verticalalignment="bottom")
        plt.text(z_lyb,  ymax, r"Z_OI",   rotation=90, verticalalignment="bottom")
        plt.text(z_oi,   ymax, r"Z_OI",   rotation=90, verticalalignment="bottom")
        plt.text(z_siiv, ymax, r"Z_SIIV", rotation=90, verticalalignment="bottom")
        plt.text(z_civ,  ymax, r"Z_CIV",  rotation=90, verticalalignment="bottom")

        plt.text(self.z_true[nspec],  ymax, r"Z_QSO",  rotation=90, verticalalignment="bottom")

        plt.xlabel("z samples")
        plt.ylabel("posteriors")
        plt.legend()

    def plot_this_mu(self, nspec, suppressed=True, num_voigt_lines=3, num_forest_lines=6, 
            label="", new_fig=True, color="red", z_sample=None):
        '''
        Plot the spectrum with the dla model

        Parameters:
        ----
        nspec (int) : index of the spectrum in the catalogue
        suppressed (bool) : apply Lyman series suppression to the mean-flux or not
        num_voigt_lines (int, min=1, max=31) : how many members of Lyman series in the DLA Voigt profile
        number_forest_lines (int) : how many members of Lymans series considered in the froest

        z_sample (float) : predicted z_QSO; if None, assumed using z_map

        Returns:
        ----
        rest_wavelengths : rest wavelengths for the DLA model
        this_mu : the DLA model
        '''
        # spec id
        plate, mjd, fiber_id = (self.plates[nspec], self.mjds[nspec], self.fiber_ids[nspec])

        # for obs data
        this_wavelengths    = self.find_this_wavelengths(nspec)
        this_noise_variance = self.find_this_noise_variance(nspec)
        this_flux           = self.find_this_flux(nspec)

        # make the choice of z_qso flexible to allow visual inspecting the fitted spectra
        if z_sample:
            z_qso = z_sample
        else:
            z_qso = self.z_map[nspec]

        this_rest_wavelengths = this_wavelengths / ( 1 + z_qso )

        # for Z code, the normalisation is not included in the preload
        this_flux, this_noise_variance = self.normalisation(
            this_rest_wavelengths, this_flux, this_noise_variance,
            self.normalization_min_lambda, self.normalization_max_lambda)

        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, z_qso)

        # for building GP model
        rest_wavelengths = self.GP.rest_wavelengths
        this_mu          = self.GP.mu

        # count the effective optical depth from members in Lyman series
        scale_factor = self.total_scale_factor(
            self.GP.tau_0_kim, self.GP.beta_kim, z_qso, 
            self.GP.rest_wavelengths, num_lines=num_forest_lines)

        # construct the uncertainty, diag(K) + Omega + V
        # build covariance matrix
        v_interpolator = interp1d(this_rest_wavelengths, this_noise_variance)

        # only take noise in between observed lambda
        ind =  (this_rest_wavelengths.min() <= rest_wavelengths) & (
                this_rest_wavelengths.max() >= rest_wavelengths)

        this_v = v_interpolator(rest_wavelengths[ind])

        this_omega2 = np.exp( 2 * self.GP.log_omega )

        # scaling factor in the omega2
        noise_scale_factor = self.total_scale_factor(
            np.exp(self.GP.log_tau_0), np.exp(self.GP.log_beta), z_qso,
            self.GP.rest_wavelengths, num_lines=num_forest_lines)
        noise_scale_factor = 1 - noise_scale_factor + np.exp(self.GP.log_c_0)

        this_omega2 = this_omega2 * noise_scale_factor**2

        # now we also consider the suppression in the noise
        this_M = self.GP.M
        if suppressed:
            this_mu     = this_mu     * scale_factor
            this_M      = this_M      * scale_factor[:, None]
            this_omega2 = this_omega2 * scale_factor**2.

        # you get only diag since there's no clear way to plot covariance in the plot
        K      = np.matmul(this_M , this_M.T )
        this_k = np.diag(K)

        this_error = this_omega2[ind] + this_k[ind] + this_v

        # plt.figure(figsize=(16, 5))
        if new_fig:
            make_fig()
            plt.plot(this_rest_wavelengths, this_flux, label="observed flux; spec-{}-{}-{}".format(plate, mjd, fiber_id), color="C0")

        plt.plot(rest_wavelengths, this_mu, 
            label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=0) + ": {:.3g}".format(self.p_no_dlas[nspec]), 
            color=color)

        plt.fill_between(rest_wavelengths[ind],
            this_mu[ind] - 2*this_error, this_mu[ind] + 2*this_error, alpha=0.8, color="orange")

        plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
        plt.ylabel(r"normalised flux")
        plt.legend()
        
        return rest_wavelengths, this_mu

    @staticmethod
    def normalisation(rest_wavelengths, flux, noise_variance, 
            normalization_min_lambda, normalization_max_lambda):
        '''
        Since the normalisation is no longer done in the preload_qso.m,
        we need to make another function to do normalisation

        Parameters:
        ----
        wavelengths      (np.ndarray)
        flux             (np.ndarray)
        noise_variance   (np.ndarray)
        normalization_min_lambda (float)
        normalization_max_lambda (float)

        Returns:
        ----
        flux (np.ndarray)
        noise_variance (np.ndarray)
        '''
        #normalizing here
        ind = ( (rest_wavelengths >= normalization_min_lambda) & 
            (rest_wavelengths <= normalization_max_lambda))

        this_median    = np.nanmedian(flux[ind])
        flux           = flux / this_median
        noise_variance = noise_variance / this_median ** 2

        return (flux, noise_variance)
