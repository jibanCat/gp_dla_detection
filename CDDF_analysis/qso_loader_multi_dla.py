'''
QSOLoader for multi-dla catalogue
'''
import numpy as np 
from collections import namedtuple, Counter
from scipy import integrate
from scipy.special import logsumexp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from .qso_loader import QSOLoader, make_fig
from .set_parameters import *
from .voigt import Voigt_absorption


class QSOLoaderMultiDLA(QSOLoader):
    '''
    A specific QSOLoader for max_dlas > 1
    '''
    def __init__(self, preloaded_file="preloaded_qsos.mat", catalogue_file="catalog.mat", 
            learned_file="learned_qso_model_dr9q_minus_concordance.mat", processed_file="processed_qsos_dr12q.mat",
            dla_concordance="dla_catalog", los_concordance="los_catalog", snrs_file="snrs_qsos_dr12q.mat",
            sub_dla=True, sample_file="dla_samples.mat", occams_razor=1):
    
        super().__init__(preloaded_file, catalogue_file, learned_file, processed_file,
            dla_concordance, los_concordance, snrs_file, sub_dla, sample_file, occams_razor)

    @staticmethod
    def make_array_multi(num_dla, array):
        '''
        make an array of los to an array with num_dla sub-los. 
        Can be imagined as splitting a single los to num_dla pieces.
        '''
        assert num_dla > 1
        return (np.ones(num_dla)[:, None] * array).ravel()

    def make_multi_unique_id(self, num_dla, plates, mjds, fiber_ids):
        '''
        To count the multi-dlas, it is better to count on the basis of number of 
        dlas instead of number of sightlines. If we count on the number of dlas,
        we will not intrepret a DLA(2) model as a single false positive detection if 
        the truth is DLA(1). Instead, we will have one true positive and one false positive.

        '''
        multi_unique_id = np.ones(num_dla)[:, None] * self.make_unique_id(plates, mjds, fiber_ids)
        assert np.prod(multi_unique_id.shape) == num_dla * plates.shape[0]
        return multi_unique_id.ravel()

    def make_multi_ROC(self, catalog):
        '''
        Make a ROC curve with a given `catalog`, which must contains 
        (num_dla, multi_real_index_dla, multi_real_index_los).

        multi_real_index should be the real index for any arrays passing through 
        self.make_array_multi(num_dla, array)
        '''
        # bool array, this builds an array describing sub-los containing DLAs or not
        # Note: real_index are unique, non-repetitive
        dla_ind = np.in1d( catalog.multi_real_index_los, catalog.multi_real_index_dla ) # shape : flatten( (num_dla, num_los) )

        # treat each single element as one sightline
        multi_model_posteriors_dla = self.model_posteriors_dla.ravel()
        
        p_dlas = multi_model_posteriors_dla[catalog.multi_real_index_los]
        p_no_dlas = self.multi_p_no_dlas.ravel()[catalog.multi_real_index_los]
        
        odds_dla_no_dla = p_dlas / p_no_dlas

        rank_idx = np.argsort( odds_dla_no_dla ) # small odds -> large odds
        assert p_dlas[ rank_idx[0] ] < p_dlas[ rank_idx[-1] ]

        # re-order every arrays based on the rank
        dla_ind         = dla_ind[ rank_idx ]
        odds_dla_no_dla = odds_dla_no_dla[ rank_idx ]

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

    def _get_parks_estimations(self, dla_parks, p_thresh=0.98, prior=False):
        '''
        Get z_dlas and log_nhis from Parks' (2018) estimations
        '''
        if 'dict_parks' not in dir(self):
            self.dict_parks = self.prediction_json2dict(dla_parks)

        if 'p_thresh' in self.dict_parks.keys():
            if self.dict_parks['p_thresh'] == p_thresh:
                unique_ids  = self.dict_parks['unique_ids']
                log_nhis    = self.dict_parks['cddf_log_nhis']  
                z_dlas      = self.dict_parks['cddf_z_dlas']    
                min_z_dlas  = self.dict_parks['min_z_dlas']
                max_z_dlas  = self.dict_parks['max_z_dlas']
                snrs        = self.dict_parks['snrs']      
                all_snrs    = self.dict_parks['all_snrs']  
                p_dlas      = self.dict_parks['cddf_p_dlas']    
                p_thresh    = self.dict_parks['p_thresh']  

                return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas

        dict_parks = self.dict_parks

        # construct an array of unique ids for los
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'] ) 
        assert unique_ids.dtype is np.dtype('int64')
        assert self.unique_ids.dtype is np.dtype('int64')

        # fixed range of the sightline ranging from 911A-1215A in rest-frame
        # we should include all sightlines in the dataset        
        roman_inds = np.isin(unique_ids, self.unique_ids)

        z_qsos     = dict_parks['z_qso'][roman_inds]
        uids       = unique_ids[roman_inds]
        
        uids, indices = np.unique( uids, return_index=True )

        # for loop to get snrs from sbird's snrs file
        all_snrs           = np.zeros( uids.shape )

        for i,uid in enumerate(uids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            all_snrs[i]           = self.snrs[real_index]

        z_qsos     = z_qsos[indices]

        min_z_dlas = (1 + z_qsos) *  lyman_limit  / lya_wavelength - 1
        max_z_dlas = (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1

        # get DLA properties
        # note: the following indices are DLA-only
        dla_inds = dict_parks['dla_confidences'] > 0.005 # use p_thresh=0.005 to filter out non-DLA spectra and 
                                                         # speed up the computation

        unique_ids = unique_ids[dla_inds]
        log_nhis   = dict_parks['log_nhis'][dla_inds]
        z_dlas     = dict_parks['z_dlas'][dla_inds]
        z_qsos     = dict_parks['z_qso'][dla_inds]
        p_dlas     = dict_parks['dla_confidences'][dla_inds]

        # check if all ids are in Roman's sample
        roman_inds = np.isin(unique_ids, self.unique_ids)
        unique_ids = unique_ids[roman_inds]
        log_nhis   = log_nhis[roman_inds]
        z_dlas     = z_dlas[roman_inds]
        z_qsos     = z_qsos[roman_inds]
        p_dlas     = p_dlas[roman_inds]

        # for loop to get snrs from sbird's snrs file
        snrs           = np.zeros( unique_ids.shape )
        log_priors_dla = np.zeros( unique_ids.shape )

        for i,uid in enumerate(unique_ids):
            real_index = np.where( self.unique_ids == uid )[0][0]

            snrs[i]           = self.snrs[real_index]
            log_priors_dla[i] = self.log_priors_dla[real_index]

        # re-calculate dla_confidence based on prior of DLAs given z_qsos
        if prior:
            p_dlas = p_dlas * np.exp(log_priors_dla)
            p_dlas = p_dlas / np.max(p_dlas)

        dla_inds = p_dlas > p_thresh

        unique_ids     = unique_ids[dla_inds]
        log_nhis       = log_nhis[dla_inds]
        z_dlas         = z_dlas[dla_inds]
        z_qsos         = z_qsos[dla_inds]
        p_dlas         = p_dlas[dla_inds]
        snrs           = snrs[dla_inds]
        log_priors_dla = log_priors_dla[dla_inds]

        # get rid of z_dlas larger than z_qsos or lower than lyman limit
        z_cut_inds = (
            z_dlas > ((1 + z_qsos) *  lyman_limit  / lya_wavelength - 1) ) 
        z_cut_inds = np.logical_and(
            z_cut_inds, (z_dlas < ( (1 + z_qsos) *  lya_wavelength  / lya_wavelength - 1 )) )

        unique_ids     = unique_ids[z_cut_inds]
        log_nhis       = log_nhis[z_cut_inds]
        z_dlas         = z_dlas[z_cut_inds]
        z_qsos         = z_qsos[z_cut_inds]
        p_dlas         = p_dlas[z_cut_inds]
        snrs           = snrs[z_cut_inds]
        log_priors_dla = log_priors_dla[z_cut_inds]

        # # for loop to get min z_dlas and max z_dlas search range from processed data
        # min_z_dlas = np.zeros( unique_ids.shape )
        # max_z_dlas = np.zeros( unique_ids.shape )

        # for i,uid in enumerate(unique_ids):
        #     real_index = np.where( self.unique_ids == uid )[0][0]

        #     min_z_dlas[i] = self.min_z_dlas[real_index]
        #     max_z_dlas[i] = self.max_z_dlas[real_index]

        # # Parks chap 3.2: fixed range of the sightline ranging from 900A-1346A in rest-frame
        # min_z_dlas = (1 + z_qsos) *  900   / lya_wavelength - 1
        # max_z_dlas = (1 + z_qsos) *  1346  / lya_wavelength - 1

        # assert np.all( ( z_dlas < max_z_dlas[0] ) & (z_dlas > min_z_dlas[0]) )

        self.dict_parks['unique_ids']    = unique_ids
        self.dict_parks['cddf_log_nhis'] = log_nhis
        self.dict_parks['cddf_z_dlas']   = z_dlas
        self.dict_parks['min_z_dlas']    = min_z_dlas 
        self.dict_parks['max_z_dlas']    = max_z_dlas
        self.dict_parks['snrs']          = snrs
        self.dict_parks['all_snrs']      = all_snrs
        self.dict_parks['cddf_p_dlas']   = p_dlas
        self.dict_parks['p_thresh']      = p_thresh

        return unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas

    def plot_cddf_parks(
            self, dla_parks, zmin=1., zmax=6., label='Parks', color=None, moment=False, 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        plot the column density function of Parks' (2018) catalogue
        '''
        (l_N, cddf, xerrs) = self.column_density_function_parks(
            dla_parks, z_min=zmin, z_max=zmax, p_thresh=p_thresh, 
            snr_thresh=snr_thresh, prior=prior, apply_p_dlas=apply_p_dlas)

        if moment:
            cddf *= 10**l_N

        plt.errorbar(10**l_N, cddf, xerr=(xerrs[0], xerrs[1]), fmt='o', label=label, color=color)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel(r"$N_\mathrm{HI}$ (cm$^{-2}$)")
        plt.ylabel(r"$f(N_\mathrm{HI})$")
        return (l_N, cddf)
        
    def column_density_function_parks(
            self, dla_parks, z_min=1., z_max=6., lnhi_nbins=30, lnhi_min=20., lnhi_max=23., 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        Compute the column density function for Parks' catalogue.
        The column density function if the number of absorbers per 
        sightlines with a given column density interval [NHI, NHI + dNHI]
        at a given absorption distance:

        f(N) = d n_DLA / dN / dX 
        or
        f(N) = n_DLA / ΔN / ΔX where n_DLA is the number of DLA
        in a given bin.

        Parameters:
        ----
        dla_parks (str)  : path to the filename of Parks prediction_DR12.json 
        z_min (float)
        z_max (float)
        lnhi_nbins (int) : number of bins in log NHI
        lnhi_min (float)
        lnhi_max (float)    

        Returns:
        ----
        (l_Ncent, cddf, xerrs)

        Note:
        ----
        See also:
        rmgarnett/gp_dla_detection/CDDF_analysis/calc_cddf.py
        sbird/fake_spectra/spectra.py
        '''
        # log NHI bins
        lnhis = np.linspace(lnhi_min, lnhi_max, num=lnhi_nbins + 1)

        # get Parks' point estimations
        unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas = self._get_parks_estimations(
            dla_parks, p_thresh=p_thresh, prior=prior)

        # filter based on snr threshold
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs > snr_thresh

        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired samples 
        inds = (log_nhis > lnhi_min) * (log_nhis < lnhi_max) * (z_dlas < z_max) * (z_dlas > z_min)
        inds = np.logical_and( snr_inds, inds )

        log_nhis   = log_nhis[inds]
        z_dlas     = z_dlas[inds]
        p_dlas     = p_dlas[inds]

        # ref: https://github.com/sbird/fake_spectra/blob/master/fake_spectra/spectra.py#L976
        if apply_p_dlas:
            tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis, weights=p_dlas)
        else:
            tot_f_N, NHI_table = np.histogram(10**log_nhis, 10**lnhis)

        dX = self.path_length(min_z_dlas, max_z_dlas, z_min, z_max)
        dN = np.array( [10**lnhi_x - 10**lnhi_m for (lnhi_m, lnhi_x) in zip( lnhis[:-1], lnhis[1:] )] )

        cddf = tot_f_N / dX / dN

        l_Ncent = np.array([ (lnhi_x + lnhi_m) / 2. for (lnhi_m, lnhi_x) in zip(lnhis[:-1], lnhis[1:]) ])
        xerrs = (10**l_Ncent - 10**lnhis[:-1], 10**lnhis[1:] - 10**l_Ncent)
        
        return (l_Ncent, cddf, xerrs)

    def plot_line_density_park(
            self, dla_parks, zmin=2, zmax=4, label="Parks", color=None, 
            p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        plot the line density as a function of redshift
        '''
        z_cent, dNdX, xerrs = self.line_density_park(
            dla_parks, z_min=zmin, z_max=zmax, 
            p_thresh=p_thresh, snr_thresh=snr_thresh, prior=prior, apply_p_dlas=apply_p_dlas)

        plt.errorbar(z_cent, dNdX, xerr=xerrs, fmt='o', label=label, color=color)
        plt.xlabel(r'z')
        plt.ylabel(r'dN/dX')
        plt.xlim(zmin, zmax)
        return z_cent, dNdX

    def line_density_park(
            self, dla_parks, z_min=2, z_max=4, lnhi_min=20.3, 
            bins_per_z=6, p_thresh=0.98, snr_thresh=-2, prior=False, apply_p_dlas=False):
        '''
        Compute the line density, the total number of DLA slightlines divided by
        the total number of sightlines, multiplied by dL/dX,
        which is dN/dX = l_DLA(z)
        '''
        nbins = np.max([ int( (z_max - z_min) * bins_per_z ), 1])
        
        # get the redshift bins
        z_bins = np.linspace(z_min, z_max, nbins + 1)

        # get Parks' point estimations
        unique_ids, log_nhis, z_dlas, min_z_dlas, max_z_dlas, snrs, all_snrs, p_dlas = self._get_parks_estimations(
            dla_parks, p_thresh=p_thresh, prior=prior)

        # filter based on snr threshold
        all_snr_inds = all_snrs > snr_thresh
        snr_inds     = snrs > snr_thresh

        min_z_dlas = min_z_dlas[all_snr_inds]
        max_z_dlas = max_z_dlas[all_snr_inds]

        # desired DLA samples 
        inds = (log_nhis > lnhi_min)
        inds = np.logical_and( snr_inds, inds )

        z_dlas     = z_dlas[inds]
        p_dlas     = p_dlas[inds]

        # point estimate of number of DLAs
        if apply_p_dlas:
            ndlas, _ = np.histogram( z_dlas, z_bins, weights=p_dlas )
        else:
            ndlas, _ = np.histogram( z_dlas, z_bins )

        # calc dX for z_bins
        dX = np.array([ self.path_length(min_z_dlas, max_z_dlas, z_m, z_x) 
            for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ])

        ii = np.where( dX > 0)
        dX = dX[ii]

        dNdX = ndlas[ii] / dX

        z_cent = np.array( [ (z_x + z_m) / 2. for (z_m, z_x) in zip(z_bins[:-1], z_bins[1:]) ] )
        xerrs  = (z_cent[ii] - z_bins[:-1][ii], z_bins[1:][ii] - z_cent[ii])

        return (z_cent[ii], dNdX, xerrs)

    def make_multi_confusion(
            self, catalog, dla_confidence=0.98, p_thresh=0.98, hard_cut=False,
            snr=-1, lyb=False, min_log_nhi=20.3):
        r'''make a confusion matrix for multi-DLA classification

        Parameters:
        ----
        catalog (collections.namedtuple) : assume to be Parks' catalog, but 
            could be extend to other catalogs
        snr : only compare spectra with signal-to-noise > snr
        lyb : only compare DLAs with z_DLA > z_QSO  

        What we want is to compute a confusion matrix for multi-DLAs such as:
        
        Garnett \ Parks    no DLA  1DLA    2DLAs   3DLAs
        ----------------------------------------------
        no DLA
        1 DLA
        2 DLAs
        3 DLAs    
        '''
        # Generally, we can loop over the entries in the table one-by-one and 
        # count the numbers on the matrix.
        # It's possible put a hard cut on the model posterior MDLA(k) to drop
        # all the spectra with max(MDLA(k)) < p_thresh, but it will potentially
        # lost lots of spectra we could count in our statistics. 
        # We thus propose to re-evaluate the model_posterior until we get 
        # MDLA(k >= num_dlas) > p_thresh.
        
        # we want unique_ids | Garnett ∩ Parks
        # TODO: generalize this to allow concordance

        # we need to introduce 
        inds = np.isin(self.unique_ids, catalog.raw_unique_ids)
        
        # SNRs cutoff
        inds = inds * ( self.snrs > snr) 

        # Garnett -> Garnett ∩ Parks
        unique_ids = self.unique_ids[inds]
        z_qsos     = self.z_qsos[inds]

        # initialize confusion matrix
        size = self.model_posteriors.shape[1] - self.sub_dla
        confusion_matrix = np.zeros((size, size))

        # matrix to store uid and DLA predictions between Garnett and Parks
        # (uid, # DLAs in Garnett, # DLAs in Parks)
        uid_matrix = np.zeros((len(unique_ids), 3)).astype(np.long)

        for i,uid in enumerate(unique_ids):

            # garnett counts
            inds_g = (uid == self.unique_ids)
            this_model_posteriors = self.model_posteriors[inds_g]
            this_map_z_dlas       = self.map_z_dlas[inds_g][0]
            this_map_log_nhis     = self.map_log_nhis[inds_g][0]
            assert this_model_posteriors.shape[0] == 1

            # z_dla cutoff and lognhi cutoff
            if lyb:
                min_z_dla = (1 + z_qsos[i]) * lyb_wavelength / lya_wavelength - 1
            else:
                min_z_dla = 0

            # put the cutoff of z_dla and lognhi in the query
            n = self.query_least_num_dlas(this_model_posteriors[0], p_thresh)
            if n > 0:
                inds = (this_map_z_dlas[n - 1, :] > min_z_dla) * (this_map_log_nhis[n - 1, :] > min_log_nhi)
                n = np.sum(inds)

            # parks counts
            m = self.query_least_num_dlas_parks(
                uid, catalog.raw_unique_ids, catalog.raw_dla_confidences, catalog.raw_z_dlas, catalog.raw_log_nhis, 
                dla_confidence, min_z_dla, min_log_nhi)

            uid_matrix[i, 0] = uid
            uid_matrix[i, 1] = n 
            uid_matrix[i, 2] = m

            if m >= size:
                m = size - 1 # max size fix to Garnett's size
            confusion_matrix[n, m] += 1


        # hard-cut
        return confusion_matrix, uid_matrix

    @staticmethod
    def downward_model(posteriors):
        '''
        Remove the final model and re-evaluate the posteriors
        '''
        return posteriors[:-1] / np.sum( posteriors[:-1] )

    def query_least_num_dlas(self, this_model_posteriors, p_thresh):
        '''
        Get the least num_dlas based on the given p_thresh.
        It will loop over from the largest num_dlas in the 
        model. It will stop at the MDLA(k) which has probability
        larger than p_thresh.
        If it couldn't find any P(DLA(k)) > p_thresh, then return
        zero DLAs.
        '''
        tot_num_dlas = len(this_model_posteriors) - 1 - self.sub_dla

        # start with the last posterior
        for i in range(tot_num_dlas):
            posterior = this_model_posteriors[::-1][0]

            if posterior > p_thresh:
                return tot_num_dlas - i

            this_model_posteriors = self.downward_model(this_model_posteriors)

        # if finding no P(DLA(>k)) larger than p_thresh, return no DLA 
        return 0

    @staticmethod
    def query_least_num_dlas_parks(
            uid, raw_unique_ids, raw_dla_confidences, raw_z_dlas, raw_log_nhis, 
            dla_confidence, min_z_dla, min_log_nhi):
        '''
        Get the (num_dlas | Parks, p_dla > dla_confidence, z_dla > min_z_dla, log_nhi > min_log_nhi)
        '''
        # get all the unique_ids corresponding to the given uid
        inds = ( uid == raw_unique_ids )

        # count the number of ids which have p_dla > dla_confidence
        return np.sum(
            (raw_dla_confidences[inds] > dla_confidence) * 
            (raw_z_dlas[inds] > min_z_dla) * 
            (raw_log_nhis[inds] > min_log_nhi)
            )

    def load_dla_parks(self, dla_parks, p_thresh=0.5, release='dr12', multi_dla=True, num_dla=2):
        '''
        load predictions_DR12.json from Parks(2018) catalogue

        Also, matched the existed thing_ids in the test data.
        Note: we have to consider DLAs from the same sightlines as different objects

        Parameters:
        ----
        dla_parks (str) : the filename of Parks (2018) product
        p_thresh (float): the minimum probability to be considered as a DLA in Parks(2018)
        release (str) 
        multi_dla (bool): whether or not we want to construct multi-dla index
        num_dla (int)   : number of dla we want to consider if we are considering multi-dlas

        Note:
        ---
        unique_ids (array) : plates * 10**9 + mjds * 10**4 + fiber_ids,
            this is an unique array constructed for matching between Parks and Roman's catalogues.
            note that make sure it is int64 since we have exceeded 2**32
        '''
        dict_parks = self.prediction_json2dict(dla_parks)

        # construct an array of unique ids for los
        self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
        unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  
        assert unique_ids.dtype is np.dtype('int64')
        assert self.unique_ids.dtype is np.dtype('int64')

        # TODO: make the naming of variables more consistent
        parks_in_garnett_inds = np.in1d( unique_ids, self.unique_ids )
        raw_unique_ids      = unique_ids[parks_in_garnett_inds]
        raw_z_dlas          = dict_parks['z_dlas'][parks_in_garnett_inds]
        raw_log_nhis        = dict_parks['log_nhis'][parks_in_garnett_inds]
        raw_dla_confidences = dict_parks['dla_confidences'][parks_in_garnett_inds]

        real_index_los = np.where( np.in1d(self.unique_ids, unique_ids) )[0]
        
        unique_ids_los = self.unique_ids[real_index_los]
        thing_ids_los  = self.thing_ids[real_index_los]
        assert np.unique(unique_ids_los).shape[0] == unique_ids_los.shape[0] # make sure we don't double count los

        # construct an array of unique ids for dlas
        dla_inds = dict_parks['dla_confidences'] > p_thresh

        real_index_dla = np.where( np.in1d(self.unique_ids, unique_ids[dla_inds]) )[0] # Note that in this step we lose
                                                                              # the info about multi-DLA since 
                                                                              # we are counting based on los

        unique_ids_dla = self.unique_ids[real_index_dla]
        thing_ids_dla  = self.thing_ids[real_index_dla]

        # Construct a list of sub-los index and dla detection based on sub-los.
        # This is a relatively complicate loop and it's hard to understand philosophically.
        # It's better to write an explaination in the paper.
        if multi_dla:
            self.multi_unique_ids = self.make_multi_unique_id(num_dla, self.plates, self.mjds, self.fiber_ids) 
            multi_unique_ids      = self.make_multi_unique_id(
                num_dla, dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  # note here some index repeated 
                                                                                             # more than num_dla times

            multi_real_index_los = np.where( np.in1d(self.multi_unique_ids, multi_unique_ids) )[0] # here we have a real_index array
                                                                                                 # exactly repeat num_dla times

            multi_unique_ids_los = self.multi_unique_ids[multi_real_index_los]

            self.multi_thing_ids = self.make_array_multi(num_dla, self.thing_ids)
            multi_thing_ids_los  = self.multi_thing_ids[multi_real_index_los]

            # loop over unique_ids to assign DLA detection to sub-los
            # Note: here we ignore the z_dla of DLAs.
            dla_multi_inds = np.zeros(multi_unique_ids_los.shape, dtype=bool)
            for uid in np.unique(multi_unique_ids_los):
                k_dlas = ( dict_parks['dla_confidences'][unique_ids == uid] > p_thresh ).sum()

                k_dlas_val = np.zeros(num_dla, dtype=bool)
                k_dlas_val[:k_dlas] = True                 # assigning True until DLA(k)

                # assign DLA detections to the unique_ids of sub-los
                dla_multi_inds[ multi_unique_ids_los == uid ] = k_dlas_val
                assert multi_unique_ids_los[ multi_unique_ids_los == uid ].shape[0] == num_dla
                
            multi_real_index_dla = multi_real_index_los[dla_multi_inds]
            multi_unique_ids_dla = multi_unique_ids_los[dla_multi_inds]
            multi_thing_ids_dla  = multi_thing_ids_los[dla_multi_inds]

            # store data in named tuple under self
            dla_catalog = namedtuple(
                'dla_catalog_parks', 
                ['real_index', 'real_index_los', 
                'thing_ids', 'thing_ids_los',
                'unique_ids', 'unique_ids_los',
                'multi_real_index_dla', 'multi_real_index_los',
                'multi_thing_ids_dla', 'multi_thing_ids_los',
                'multi_unique_ids_dla', 'multi_unique_ids_los', 
                'release', 'num_dla', 
                'raw_unique_ids', 'raw_z_dlas', 'raw_log_nhis', 'raw_dla_confidences' ])
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla, real_index_los=real_index_los, 
                thing_ids=thing_ids_dla, thing_ids_los=thing_ids_los, 
                unique_ids=unique_ids_dla, unique_ids_los=unique_ids_los,
                multi_real_index_dla=multi_real_index_dla, multi_real_index_los=multi_real_index_los,
                multi_thing_ids_dla=multi_thing_ids_dla, multi_thing_ids_los=multi_thing_ids_los,
                multi_unique_ids_dla=multi_unique_ids_dla, multi_unique_ids_los=multi_unique_ids_los,
                release=release, num_dla=num_dla,
                raw_unique_ids=raw_unique_ids, raw_z_dlas=raw_z_dlas, 
                raw_log_nhis=raw_log_nhis, raw_dla_confidences=raw_dla_confidences)

        else:
            dla_catalog = namedtuple(
                'dla_catalog_parks', 
                ['real_index', 'real_index_los', 
                'thing_ids', 'thing_ids_los',
                'unique_ids', 'unique_ids_los',
                'release',
                'raw_unique_ids', 'raw_z_dlas', 'raw_log_nhis', 'raw_dla_confidences' ])
            self.dla_catalog_parks = dla_catalog(
                real_index=real_index_dla, real_index_los=real_index_los, 
                thing_ids=thing_ids_dla, thing_ids_los=thing_ids_los, 
                unique_ids=unique_ids_dla, unique_ids_los=unique_ids_los,
                release=release,
                raw_unique_ids=raw_unique_ids, raw_z_dlas=raw_z_dlas, 
                raw_log_nhis=raw_log_nhis, raw_dla_confidences=raw_dla_confidences)


    def make_MAP_parks_comparison(self, catalog, num_dlas=1, dla_confidence=0.98):
        '''
        make a comparison between map values and Park's predictions

        What really computed is:
             (map_z_dla - z_dla_parks    | Parks DLA(1 to n) ∩ garnett DLA(1 to n) ) and
             (map_log_nhi - log_nhi_parks| Parks DLA(1 to n) ∩ garnett DLA(1 to n) ),
        which means
             we only compares spectra containing the same number of DLAs        
        '''
        # Parks contains multiple DLAs, which means we have to identify a way to compare multiple-DLAS
        # within one spectroscopic observations. 
        # We only compare Parks(DLA==n) ∩ Garnett(DLA==n) where n is num_dlas detected within a sightline.
        # The purpose is to test the systematic of mutli-DLA models, though we know Parks have intrinsic bias
        # in estimations of NHI.
        # 
        # We first find indices for Parks(DLA==n) and 
        #     then find indices for Garnett(DLA==n)
        # for multi-DLAs, we find the minimum z_dla value of difference between Parks and Garnett, which is
        #     min( z_dlas[ith qso, Parks(DLA==n)] - z_dlas[ith qso, Garnett(DLA==n)] )
        # while for column density NHI, we use the the DLAs corresponding to minimum z_dlas to compute the MAP difference.
        assert num_dlas > 0

        # filter out low confidence DLAs
        raw_unique_ids  = catalog.raw_unique_ids
        dla_confidences = catalog.raw_dla_confidences
        raw_z_dlas      = catalog.raw_z_dlas
        raw_log_nhis    = catalog.raw_log_nhis

        raw_unique_ids  = raw_unique_ids[dla_confidences > dla_confidence]
        raw_z_dlas      = raw_z_dlas[dla_confidences > dla_confidence]
        raw_log_nhis    = raw_log_nhis[dla_confidences > dla_confidence]

        count_unique_ids = Counter(raw_unique_ids)
        inds  = np.array(list(count_unique_ids.values()), dtype=np.int) == num_dlas
        
        # find IDs | Parks(DLA==n)
        uids_dla_n_parks   =  np.array(list(count_unique_ids.keys()), dtype=np.int)[inds]
        
        # find IDs | Garnett(DLA==n)
        uids_dla_n_garnett = self.unique_ids[self.dla_map_num_dla == num_dlas]
        
        # find IDs | Parks(DLA==n) ∩ Garnett(DLA==n)
        # Note: the intersection for DLA(n==num_dlas) between Parks and Garnett is surprising small.  
        uids_dla_n = uids_dla_n_parks[np.isin(uids_dla_n_parks, uids_dla_n_garnett)]

        inds_dla_n = np.isin( raw_unique_ids, uids_dla_n )

        # for each ID, it has num_dlas elements in the following arrays
        z_dlas_parks     = raw_z_dlas[inds_dla_n]
        log_nhis_parks   = raw_log_nhis[inds_dla_n]
        unique_ids_parks = raw_unique_ids[inds_dla_n]

        # looping over each ID and comparing the MAP values between Parks and Garnett
        Delta_z_dlas   = np.empty((len(uids_dla_n), num_dlas))
        Delta_log_nhis = np.empty((len(uids_dla_n), num_dlas))
        # TODO: to see if there is any way to avoiding using for loop, though it seems to be fine 
        # for multi-DLAs since they are rare events.
        for i,uid in enumerate(uids_dla_n):
            # find Garnett's MAPs(Parks(DLA==n) ∩ Garnett(DLA==n))
            ind = self.unique_ids == uid
            
            this_z_dlas   = self.map_z_dlas[ind][0, num_dlas - 1, :num_dlas]
            this_log_nhis = self.map_log_nhis[ind][0, num_dlas - 1, :num_dlas]

            # find Parks' predictions(Parks(DLA==n) ∩ Garnett(DLA==n))
            ind_parks = unique_ids_parks == uid
            this_z_dlas_parks   = z_dlas_parks[ind_parks]
            this_log_nhis_parks = log_nhis_parks[ind_parks]

            # sort z_dlas to minimize the difference of z_dlas between Parks and Garnett
            argsort_garnett = np.argsort(this_z_dlas) 
            argsort_parks   = np.argsort(this_z_dlas_parks)

            assert np.all(
                    np.abs((this_z_dlas[argsort_garnett] - this_z_dlas_parks[argsort_parks]).sum()) <= 
                    np.abs((this_z_dlas                  - this_z_dlas_parks).sum())
                    )

            Delta_z_dlas[i, :]   = (this_z_dlas[argsort_garnett]   - this_z_dlas_parks[argsort_parks])
            Delta_log_nhis[i, :] = (this_log_nhis[argsort_garnett] - this_log_nhis_parks[argsort_parks])

        return Delta_z_dlas, Delta_log_nhis, z_dlas_parks

    @staticmethod
    def prediction_json2dict(filename_parks="predictions_DR12.json", object_name='dlas'):
        '''
        extract dlas or subdlas or lyb 
        and convert to a dataframe

        Parameters : 
        --- 
        filename_parks (str) : predictions_DR12.json, from Parks (2018)
        object_name (str) : "dlas" or "subdlas" or "lyb"

        Return : 
        ---
        dict_parks (dict) : a dictionary contains the predictions from Parks (2018)
        '''
        import json
        with open(filename_parks, 'r') as f:
            parks_json = json.load(f)

        num_dlas = "num_{}".format(object_name)

        # extract DLA information (exclude subDLA, lyb)
        ras       = []
        decs      = []
        plates    = []
        mjds      = []
        fiber_ids = []
        z_qsos    = []
        dla_confidences = []
        z_dlas    = []
        log_nhis  = []

        for table in parks_json:
            # extract plate-mjd-fiber_id
            plate, mjd, fiber_id = table['id'].split("-")
            
            # has dla(s)
            if table[num_dlas] > 0:
                for i in range(table[num_dlas]):
                    dla_table = table[object_name][i]
                    
                    # append the basic qso info
                    ras.append(table['ra'])
                    decs.append(table['dec'])
                    plates.append(plate)       
                    mjds.append(mjd)           
                    fiber_ids.append(fiber_id) 
                    z_qsos.append(table['z_qso'])
                    
                    # append the object (dla or lyb or subdla) info
                    dla_confidences.append(dla_table['dla_confidence'])
                    z_dlas.append(dla_table['z_dla'])
                    log_nhis.append(dla_table['column_density'])
            
            # no dla
            elif table[num_dlas] == 0:
                # append basic info
                ras.append(table['ra'])
                decs.append(table['dec'])
                plates.append(plate)
                mjds.append(mjd)
                fiber_ids.append(fiber_id)
                z_qsos.append(table['z_qso'])

                # append no dla info
                dla_confidences.append(np.nan)
                z_dlas.append(np.nan)
                log_nhis.append(np.nan)
            
            else:
                print("[Warning] exception case")
                print(table)
                
        dict_parks = {
                'ras'    :          np.array(ras),
                'decs'   :          np.array(decs),
                'plates' :          np.array(plates).astype(np.int),
                'mjds'   :          np.array(mjds).astype(np.int),
                'fiber_ids' :       np.array(fiber_ids).astype(np.int),
                'z_qso'  :          np.array(z_qsos),
                'dla_confidences' : np.array(dla_confidences),
                'z_dlas' :          np.array(z_dlas),
                'log_nhis' :        np.array(log_nhis)
            }

        return dict_parks

    def plot_this_mu(self, nspec, suppressed=True, num_voigt_lines=3, num_forest_lines=31, Parks=False, dla_parks=None, 
        label="", new_fig=True, color="red"):
        '''
        Plot the spectrum with the dla model

        Parameters:
        ----
        nspec (int) : index of the spectrum in the catalogue
        suppressed (bool) : apply Lyman series suppression to the mean-flux or not
        num_voigt_lines (int, min=1, max=31) : how many members of Lyman series in the DLA Voigt profile
        number_forest_lines (int) : how many members of Lymans series considered in the froest
        Parks (bool) : whether to plot Parks' results
        dla_parks (str) : if Parks=True, specify the path to Parks' `prediction_DR12.json`

        Returns:
        ----
        rest_wavelengths : rest wavelengths for the DLA model
        this_mu : the DLA model
        map_z_dlas : MAP z_dla values 
        map_log_nhis : MAP log NHI values
        '''
        # spec id
        plate, mjd, fiber_id = (self.plates[nspec], self.mjds[nspec], self.fiber_ids[nspec])

        # for obs data
        this_wavelengths = self.find_this_wavelengths(nspec)
        this_flux        = self.find_this_flux(nspec)

        this_rest_wavelengths = emitted_wavelengths(this_wavelengths, self.z_qsos[nspec])

        # for building GP model
        rest_wavelengths = self.GP.rest_wavelengths
        this_mu = self.GP.mu

        # count the effective optical depth from members in Lyman series
        scale_factor = self.total_scale_factor(
            self.GP.tau_0_kim, self.GP.beta_kim, self.z_qsos[nspec], self.GP.rest_wavelengths, num_lines=num_forest_lines)
        if suppressed:
            this_mu = this_mu * scale_factor

        # get the MAP DLA values
        nth = np.argmax( self.model_posteriors[nspec] ) - 1 - self.sub_dla
        if nth >= 0:
            if self.model_posteriors.shape[1] > 2:
                map_z_dlas    = self.all_z_dlas[nspec, :(nth + 1)]
                map_log_nhis  = self.all_log_nhis[nspec, :(nth + 1)]
            elif self.model_posteriors.shape[1] == 2: # Garnett (2017) model
                self.prepare_roam_map_vals_per_spec(nspec, self.sample_file)
                
                map_z_dlas    = np.array([ self.all_z_dlas[nspec] ])
                map_log_nhis  = np.array([ self.all_log_nhis[nspec] ])
                assert ~np.isnan(map_z_dlas)

            for map_z_dla, map_log_nhi in zip(map_z_dlas, map_log_nhis):
                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]), 10**map_log_nhi, map_z_dla, num_lines=num_voigt_lines)

                this_mu = this_mu * absorption

        # get parks model
        if Parks:
            if not 'dict_parks' in dir(self):
                self.dict_parks = self.prediction_json2dict(dla_parks)

            dict_parks = self.dict_parks

            # construct an array of unique ids for los
            self.unique_ids = self.make_unique_id(self.plates, self.mjds, self.fiber_ids)
            unique_ids      = self.make_unique_id( dict_parks['plates'], dict_parks['mjds'], dict_parks['fiber_ids'])  
            assert unique_ids.dtype is np.dtype('int64')
            assert self.unique_ids.dtype is np.dtype('int64')

            uids = np.where( unique_ids == self.unique_ids[nspec] )[0]

            this_parks_mu = self.GP.mu * scale_factor
            dla_confidences = []
            z_dlas          = []
            log_nhis        = []
            
            for uid in uids:
                z_dla   = dict_parks['z_dlas'][uid]
                log_nhi = dict_parks['log_nhis'][uid]

                dla_confidences.append( dict_parks['dla_confidences'][uid] )
                z_dlas.append( z_dla )
                log_nhis.append( log_nhi )

                absorption = Voigt_absorption(
                    rest_wavelengths * (1 + self.z_qsos[nspec]), 10**log_nhi, z_dla, num_lines=1)

                this_parks_mu = this_parks_mu * absorption

        # plt.figure(figsize=(16, 5))
        if new_fig:
            make_fig()
            plt.plot(this_rest_wavelengths, this_flux, label="observed flux; spec-{}-{}-{}".format(plate, mjd, fiber_id), color="C0")

        if Parks:
            plt.plot(rest_wavelengths, this_parks_mu, label=r"Parks: z_dlas = ({}); lognhis=({}); p_dlas=({})".format(
                ",".join("{:.3g}".format(z) for z in z_dlas), 
                ",".join("{:.3g}".format(n) for n in log_nhis), 
                ",".join("{:.3g}".format(p) for p in  dla_confidences)), 
                color="orange")
        if nth >= 0:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=nth+1) + ": {:.3g}; ".format(
                    self.model_posteriors[nspec, 1 + self.sub_dla + nth]) + 
                    "lognhi = ({})".format( ",".join("{:.3g}".format(n) for n in map_log_nhis) ), 
                color=color)
        else:
            plt.plot(rest_wavelengths, this_mu, 
                label=label + r"$\mathcal{M}$"+r" DLA({n})".format(n=0) + ": {:.3g}".format(self.p_no_dlas[nspec]), 
                color=color)

        plt.xlabel(r"rest-wavelengths $\lambda_{\mathrm{rest}}$ $\AA$")
        plt.ylabel(r"normalised flux")
        plt.legend()
        
        if nth >= 0:
            return rest_wavelengths, this_mu, map_z_dlas, map_log_nhis            
        return rest_wavelengths, this_mu

    def generate_sub_dla_catalogue(self, outfile="predictions_sub_DLA_candidates.json"):
        '''
        Generate a catalogue for the candidates of sub-DLAs:
        we only record the sub-DLA probability and the QSO spec info for this catalogue
        
        Example template :
        ----
        [
            {
                "p_sub_dla" : 0.9,
                "ra": 9.2152, 
                "snr": 1.23,
                "dec": -0.1659, 
                "plate": 3586,
                "mjd": 55181,
                "fiber_id": 16,
                "thing_id": ,
                "z_qso": 2.190
            }, 
            ...
        ]        
        '''
        # repeat the same process in self.gerenate_json_catalogue()
        # but only consider the specs with sub-DLA has the highest prob
        predictions_sub = []

        # query the quasar_ind of sub-DLA model posterior is the highest
        quasar_inds = np.where(self.dla_map_model_index == 1)[0]

        # store some arrays here to query later
        # you do the test_ind indicing after storing the HDF5 array 
        # into memory. Reading from numpy array from memory is much
        # faster than you do IO from the file.
        ras  = self.catalogue_file['ras'][0, :][self.test_ind]
        decs = self.catalogue_file['decs'][0, :][self.test_ind]

        for i in quasar_inds:
            this_sub_dla = {}

            this_sub_dla['p_sub_dla'] = self.model_posteriors[i, 1].item()
            this_sub_dla['ra']        = ras[i].item()
            this_sub_dla['snr']       = self.snrs[i].item()
            this_sub_dla['dec']       = decs[i].item()
            this_sub_dla['plate']     = self.plates[i].item()
            this_sub_dla['mjd']       = self.mjds[i].item()
            this_sub_dla['fiber_id']  = self.fiber_ids[i].item()
            this_sub_dla['thing_id']  = self.thing_ids[i].item()
            this_sub_dla['z_qso']     = self.z_qsos[i].item()

            predictions_sub.append(this_sub_dla)

        import json
        with open(outfile, 'w') as json_file:
            json.dump(predictions_sub, json_file, indent=2)
        
        return predictions_sub
