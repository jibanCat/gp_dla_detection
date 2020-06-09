'''
QSOLoader for plotting relevant plots for DLA detections combine with z-estimation
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from scipy.interpolate import interp1d
import h5py

from .set_parameters import *
from .qso_loader import QSOLoader, GPLoader, make_fig
from .voigt import Voigt_absorption

class QSOLoaderZDLAs(QSOLoader):
    '''
    A specific QSOLoader for Z estimation + DLA statistics
    '''
    def __init__(self, preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat", 
            learned_file="learned_qso_model_dr9q_minus_concordance.mat", processed_file="processed_qsos_dr12q.mat",
            dla_concordance="dla_catalog", los_concordance="los_catalog", sample_file="dla_samples.mat",
            occams_razor=False, small_file = True,
            normalization_min_lambda=1176, normalization_max_lambda=1256):
        self.preloaded_file = h5py.File(preloaded_file, 'r')
        self.catalogue_file = h5py.File(catalogue_file, 'r')
        self.learned_file   = h5py.File(learned_file,   'r')
        self.processed_file = h5py.File(processed_file, 'r')
        self.sample_file    = h5py.File(sample_file, 'r')

        self.occams_razor = occams_razor

        self.normalization_min_lambda = normalization_min_lambda   # range of rest wavelengths to use   Å
        self.normalization_max_lambda = normalization_max_lambda   #   for flux normalization

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
        self.z_map         = self.processed_file['z_map'][0, :]
        self.z_true        = self.processed_file['z_true'][0, :]
        self.dla_true      = self.processed_file['dla_true'][0, :]
        self.all_z_dlas    = self.processed_file['z_dla_map'][0, :]
        self.all_log_nhis  = self.processed_file['log_nhi_map'][0, :]
        self.snrs          = self.processed_file['signal_to_noise'][0, :]
        
        # memory free loading; using disk I/O to load sample posteriors
        try:
            self.sample_log_posteriors_no_dla = self.processed_file[
                'sample_log_posteriors_no_dla']
            self.sample_log_posteriors_dla    = self.processed_file[
                'sample_log_posteriors_dla']
        except KeyError as e:
            print(e)
            print("[Warning] you are loading a truncated file without sample posteriors.")

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

        self.z_qsos     = self.z_qsos[self.test_ind]
        self.snrs_cat   = self.snrs_cat[self.test_ind]

        # [Occams Razor] Update model posteriors with an additional occam's razor
        # updating: 1) model_posteriors, p_dlas, p_no_dlas
        if occams_razor != False:
            self.model_posteriors = self._occams_model_posteriors(self.model_posteriors, self.occams_razor)
            self.p_dlas    = self.model_posteriors[:, 1:].sum(axis=1)
            self.p_no_dlas = self.model_posteriors[:, :1].sum(axis=1)

        # build a MAP number of DLAs array- the number of DLA with the largest posteriors.
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

        self.z_map         = self.z_map[~nan_inds]
        self.z_true        = self.z_true[~nan_inds]
        self.dla_true      = self.dla_true[~nan_inds]
        self.all_z_dlas    = self.all_z_dlas[~nan_inds]
        self.all_log_nhis  = self.all_log_nhis[~nan_inds]

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

    def make_ROC(self, catalog, occams_razor=1):
        '''
        Make a ROC curve with a given `catalog`, which must contains (real_index, real_index_los)
        '''
        dla_ind = np.in1d( catalog.real_index_los, catalog.real_index ) # boolean array, same size

        # use log_posteriors_dla directly to avoid numerical underflow
        log_posteriors_dla    = self.processed_file['log_posteriors_dla'][0, :]       - np.log(occams_razor)
        log_posteriors_no_dla = self.processed_file['log_posteriors_no_dla'][0, :]

        # filtering out ~nan_inds
        log_posteriors_dla    = log_posteriors_dla[~self.nan_inds]
        log_posteriors_no_dla = log_posteriors_no_dla[~self.nan_inds]

        # query the corresponding index in the catalog
        log_posteriors_dla    = log_posteriors_dla[catalog.real_index_los]
        log_posteriors_no_dla = log_posteriors_no_dla[catalog.real_index_los]

        odds_dla_no_dla = log_posteriors_dla - log_posteriors_no_dla # log odds

        rank_idx = np.argsort( odds_dla_no_dla ) # small odds -> large odds
        assert log_posteriors_dla[ rank_idx[0] ] < log_posteriors_dla[ rank_idx[-1] ]

        # re-order every arrays based on the rank
        dla_ind = dla_ind[ rank_idx ]
        odds_dla_no_dla = odds_dla_no_dla[rank_idx]

        TPR = []
        FPR = []

        for odd in odds_dla_no_dla:
            odd_ind = odds_dla_no_dla >= odd
            
            true_positives   =  dla_ind &  odd_ind
            false_negatives  =  dla_ind & ~odd_ind
            true_negatives   = ~dla_ind & ~odd_ind
            false_positives  = ~dla_ind &  odd_ind

            TPR.append( np.sum(true_positives) / ( np.sum(true_positives) + np.sum(false_negatives) ) )
            FPR.append( np.sum(false_positives) / (np.sum(false_positives) + np.sum(true_negatives)) )

        return TPR, FPR

    def make_MAP_comparison(self, catalog):
        '''
        make a comparison between map values and concordance values
        
        This is (z_dla_concordance - map_z_dla | concordance ∩ garnett) and
                (log_nhi_concordance - log_nhi | concordance ∩ garnett)
        which means we only consider the difference has overlaps between concordance and ours catalogue
        '''
        # map values array size: (num_qsos, model_DLA(n), num_dlas)
        # use the real_index vals stored in dla_catalog attribute and 
        # loop over the real_index of self.map_values while at the same time
        # get the (MAP_values | DLA(n)) using self.dla_map_model_index
        real_index       = self.dla_catalog.real_index # concordance only

        # get corresponding map vals for concordance only
        map_model_index  = self.dla_map_model_index[real_index]
        
        # make sure having at least one DLA, concordance ∩ garnett
        real_index = real_index[map_model_index > 0]
        
        map_z_dlas   = self.all_z_dlas[real_index]
        map_log_nhis = self.all_log_nhis[real_index]
        
        Delta_z_dlas   = map_z_dlas   - self.dla_catalog.z_dlas[map_model_index > 0]
        Delta_log_nhis = map_log_nhis - self.dla_catalog.log_nhis[map_model_index > 0]

        return Delta_z_dlas, Delta_log_nhis

    def make_MAP_hist2d(self, p_thresh = 0.98):
        '''
        Make a 2D hist for z_map vs z_true
        p_thresh (float) : p_dlas < p_thresh are not considered to be DLAs
        '''
        real_index = self.dla_catalog.real_index # concordance only

        # get corresponding map vals for concordance only
        # and threshold the p_dlas
        this_dla_map_model_index = self.dla_map_model_index.copy()
        this_dla_map_model_index[self.p_dlas < p_thresh] = 0
        map_model_index  = this_dla_map_model_index[real_index]

        # make sure having at least one DLA, concordance ∩ garnett
        real_index = real_index[map_model_index > 0]

        map_z_dlas   = self.all_z_dlas[real_index]
        map_log_nhis = self.all_log_nhis[real_index]

        true_z_dlas   = self.dla_catalog.z_dlas[map_model_index   > 0]
        true_log_nhis = self.dla_catalog.log_nhis[map_model_index > 0]

        return map_z_dlas, true_z_dlas, map_log_nhis, true_log_nhis, real_index

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

        # this_kernel = Omega + diag(K)
        this_kernel   = this_omega2 + this_k

        # get the MAP DLA values
        nth = np.argmax( self.model_posteriors[nspec] ) - 1
        if nth >= 0:
            map_z_dlas    = np.array([ self.all_z_dlas[nspec] ])
            map_log_nhis  = np.array([ self.all_log_nhis[nspec] ])
            assert ~np.isnan(map_z_dlas)

            for map_z_dla, map_log_nhi in zip(map_z_dlas, map_log_nhis):
                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + z_qso),
                    10**map_log_nhi, map_z_dla, num_lines=num_voigt_lines)

                this_mu    = this_mu * absorption
                this_error = this_kernel * absorption**2

        this_error = this_kernel[ind] + this_v

        # plt.figure(figsize=(16, 5))
        if new_fig:
            make_fig()
            plt.plot(this_rest_wavelengths, this_flux, label="observed flux; spec-{}-{}-{}".format(plate, mjd, fiber_id), color="C0")

        if nth >= 0:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=nth+1) + ": {:.3g}; ".format(
                    self.model_posteriors[nspec, nth+1]) + 
                    "lognhi = ({})".format( ",".join("{:.3g}".format(n) for n in map_log_nhis) ), 
                color=color)
        else:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=0) + ": {:.3g}".format(self.p_no_dlas[nspec]), 
                color=color)

        plt.fill_between(rest_wavelengths[ind],
            this_mu[ind] - 2*this_error, this_mu[ind] + 2*this_error, alpha=0.8, color="orange")

        plt.xlabel(r"Rest Wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
        plt.ylabel(r"Normalized Flux")
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

    def _get_estimations(self, p_thresh=0.9):
        '''
        Get z_dlas and log_nhis from Z estimation project, ignore all uncertainties
        
        Returns:
        ----
        thing_ids  : DLA thing_ids (could have more than one DLAs)
        log_nhis   : DLA log NHI
        z_dlas     : DLA z_dlas
        min_z_dlas : minimum search absorber redshift for a given sightline
        max_z_dlas : maximum search absorber redshift for a given sightline
        all_snrs   : signal-to-noise ratios for all sightlines
        snrs       : signal-to_noise ratios for DLA sightlines (could have multiple DLAs)
        '''
        # Get the search length first, searching length dX should consider all of the sightlines
        # All slightlines:
        all_thing_ids = self.thing_ids

        z_qsos    = self.z_qsos
        all_snrs  = self.snrs

        # simply assume we search from limit to alpha
        min_z_dlas = (1 + z_qsos) * (lyman_limit + kms_to_z(3000))    / lya_wavelength - 1
        max_z_dlas = (1 + z_qsos) * (lya_wavelength - kms_to_z(3000)) / lya_wavelength - 1

        # DLA slightlines:
        dla_ind   = self.p_dlas > p_thresh

        thing_ids = all_thing_ids[dla_ind]
        z_dlas    = self.all_z_dlas[dla_ind]
        log_nhis  = self.all_log_nhis[dla_ind]
        snrs      = self.snrs[dla_ind]

        return thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs

    def column_density_function(
            self, z_min, z_max, lnhi_nbins=30, lnhi_min=20., lnhi_max=23.,
            snr_thresh=-1):
        '''
        Compute the column density distribution function for z estimation,
        ignore all uncertainties (using MAP directly)

        This should follow the convention of sbird's plot

        Note:
        ----
        See self.column_density_function_parks for more science details  
        
        Parameters:
        ----
        z_min (float) : the minimum redshift you consider to compute the CDDF
        z_max (float) : the maximum redshift you consdier to compute the CDDF
        lnhi_nbins (int) : the number of bins you put on DLA column densities
        lnhi_min (float) : the minimum log column density of DLAs you consider to plot
        lnhi_max (float) : the maximum log column density of DLAs you consdier to plot
        
        Returns:
        ----
        l_Ncent (np.ndarray) : the array of the centers of the log NHI bins
        cddf (np.ndarray)    : the CDDF you computed, which is f(N) = n_DLA / ΔN / ΔX
        xerrs (np.ndarray)   : the width of each bins you applied on the log NHI bins 
        '''
        # log NHI bins 
        lnhis = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

        # get MAP DLA detections
        thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs  = self._get_estimations()

        # SNR cut for both all sightlines and DLA slightlines
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs     > snr_thresh

        # update searching ranges from SNR cut
        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired samples
        inds = (log_nhis > lnhi_min) * (log_nhis < lnhi_max) * (z_dlas < z_max) * (z_dlas > z_min)
        
        # also update the snr cuts for the DLA sightlines
        inds = np.logical_and( snr_inds, inds )

        log_nhis = log_nhis[inds]
        z_dlas   = z_dlas[inds]

        # get CDDF from histogram
        tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis)

        dX = self.path_length(min_z_dlas, max_z_dlas, z_min, z_max)
        dN = np.array( [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip( lnhis[:-1], lnhis[1:] ) ] )

        cddf = tot_f_N / dX / dN

        l_Ncent = np.array([ (lnhi_x + lnhi_m) / 2. for (lnhi_m, lnhi_x) in zip(lnhis[:-1], lnhis[1:]) ])
        xerrs   = (10**l_Ncent - 10**lnhis[:-1], 10**lnhis[1:] - 10**l_Ncent)

        return (l_Ncent, cddf, xerrs)

    def plot_cddf(
            self, zmin=1., zmax=6., label='Z Estimation DR12', color=None, moment=False,
            snr_thresh=-1):
        '''
        plot the column density function of Z Estiamtion DR12 catalogue
        '''
        (l_N, cddf, xerrs) = self.column_density_function(
            z_min=zmin, z_max=zmax, snr_thresh=snr_thresh)

        if moment:
            cddf *= 10**l_N

        plt.errorbar(10**l_N, cddf, xerr=(xerrs[0], xerrs[1]), fmt='o', label=label, color=color)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
        plt.ylabel(r"$f(N_\mathrm{HI})$")

        return (l_N, cddf)


    def line_density(
            self, z_min=2, z_max=5, lnhi_min=20.3, bins_per_z=6,
            snr_thresh=-1):
        '''
        Compute the line density for Z Estimation DR12
        '''
        nbins = np.max([ int( (z_max - z_min) * bins_per_z ), 1 ])

        # get the redshift bins
        z_bins = np.linspace(z_min, z_max, nbins + 1) 

        # get MAP DLA detections
        thing_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas,snrs,all_snrs = self._get_estimations()

        # SNR cut for both all sightlines and DLA sightlines
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs     > snr_thresh

        # update the searching ranges using SNR cuts
        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired DLA samples
        inds = (log_nhis > lnhi_min)

        # also update the SNR cuts to the DLA slightlines
        inds = np.logical_and( snr_inds, inds )

        z_dlas = z_dlas[inds]

        # estimate the number of DLAs
        ndlas, _ = np.histogram( z_dlas, z_bins )

        dX = np.array([ self.path_length(min_z_dlas, max_z_dlas, z_m, z_x)
            for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ])

        ii = np.where( dX > 0 )
        dX = dX[ii]

        dNdX = ndlas[ii] / dX

        z_cent = np.array( [ (z_x + z_m) / 2. for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ] )
        xerrs  = (z_cent[ii] - z_bins[:-1][ii], z_bins[1:][ii] - z_cent[ii])

        return (z_cent[ii], dNdX, xerrs)

    def plot_line_density(
        self, zmin=2, zmax=5, label="Z Estimation DR12", color=None,
        snr_thresh=-1):
        '''
        plot the line density of Z Estimation DR12
        '''
        z_cent, dNdX, xerrs = self.line_density(
            z_min=zmin, z_max=zmax, snr_thresh=snr_thresh)

        plt.errorbar(z_cent, dNdX, xerr=xerrs, fmt='o', label=label, color=color)
        plt.xlabel(r'z')
        plt.ylabel(r'dN/dX')
        plt.xlim(zmin, zmax)

        return z_cent, dNdX
