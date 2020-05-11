'''
A Class wrap on top of QSOLoader to do some heavy stuff such as manipulating
the sample log likelihoods in the processed_qsos.mat. Only designed for Multi-DLA
catalogue since Garnett's (2017) catalogue is small enough to handle in QSOLoader.

This Class is expected to run in a machine with more than 60GB memory.
'''
from typing import Tuple

import numpy as np
from scipy.special import logsumexp
import h5py
from .qso_loader import QSOLoader


# utility function to do nan logmeanexp
def lognanmeanexp(array: np.ndarray) -> float:
    '''
    Do log ( nanmean( exp(array) ) )
    by using log ( sum( exp(array[~nan_ind]) ) ) - log( sum(nan_ind) )
    
    :param array: expected to be an one-dim array
    :return: a number represents the log(mean) of the input
    '''
    nan_ind = np.isnan(array)

    return logsumexp( array, b=~nan_ind ) - np.log( np.sum(nan_ind) )
    

class QSOSample(QSOLoader):
    '''
    A Class to handle sample log likelihoods but also take adavatages of
    QSOLoader's features: e.g., plotting ROC curve

    :method update_model_posteriors: update model_posteriors based on
        the DLA's logNHI criteria
    '''
    def __init__(self,
            preloaded_file: str = "preloaded_qsos.mat", catalogue_file: str = "catalog.mat", 
            learned_file: str = "learned_qso_model_dr9q_minus_concordance.mat", 
            processed_file: str = "processed_qsos_dr12q.mat",
            dla_concordance: str = "dla_catalog", los_concordance: str = "los_catalog",
            snrs_file: str=  "snrs_qsos_dr12q.mat", sample_file: str = "dla_samples.mat",
            sub_dla: bool = True, occams_razor: int = 1,
            sub_dla_sample_file: str = 'subdla_samples.mat'):
        # Inherent everthing first
        # - It will loading all necessary variables into memory, e.g.,
        #   thing_ids, model_posteriors, p_dlas, p_no_dlas, z_qsos,
        # - It will load learned model into a GP class in self.GP
        # - Concordance catalogue will automatically load into self.dla_catalog
        # - Parks catalogue will require user to run self.load_dla_parks(filename), with
        #   filename to Parks' json catalogue
        # - It will automaticall apply test_ind in the processed_file and it will also
        #   automaticall discard the empty spectra inds. So be aware that the index ordering
        #   will be different from you directly read from processed_file using MATLAB. Always 
        #   use self.thing_ids to avoid the index misalignment. And be aware the index mapping 
        #   from raw MATLAB file is stored in self.test_ind.
        #   Usage : 
        #   ----
        #   test_ind (np.ndarray, bool): mapping from catalogue_file to index in QSOLoader.
        #       self.val = self.catalogue_file[key][self.test_ind]  
        #   nan_ind  (np.ndarray, bool): mapping from processed_file to index in QSOLoader,
        #       self.val = self.processed_file[key][self.test_ind]
        super().__init__(preloaded_file, catalogue_file, learned_file, processed_file,
            dla_concordance, los_concordance, snrs_file, sub_dla, sample_file, occams_razor)

        # you must have sample likelihoods variables in your mat file
        assert "sample_log_likelihoods_dla" in self.processed_file.keys()

        # load the sample_log_likelihoods; make sure you are not using a truncated processed file
        # this is a HUGE matrix with shape (num_dlas, num_dla_samples, num_qsos)
        # load with shape (num_qsos, num_dla_samples, num_dlas)
        self.sample_log_likelihoods_dla = self.processed_file['sample_log_likelihoods_dla'][()].T
        self.sample_log_likelihoods_dla = self.sample_log_likelihoods_dla[~self.nan_inds, :, :]

        (num_qsos, num_dla_samples, num_dlas) = self.sample_log_likelihoods_dla.shape
        self.num_dlas        = num_dlas
        self.num_dla_samples = num_dla_samples

        assert num_qsos == len(self.z_qsos)

        # re-normalize based on occams razor between different DLA(k)
        # note: apply directly on sample_likelihoods here to looping over num_dlas dim
        # everytime when we want to have to evalute log_likelihoods
        # P( theta | DLA(k) ) = P( theta | DLA(k) ) * 1 / (num_dla_samples * (k - 1))
        for i in range(num_dlas):
            self.sample_log_likelihoods_dla[:, :, i] -= num_dla_samples * i # python has i = k - 1

        # another huge matrix with shape (num_dla_samples, num_qsos)
        # load with shape (num_qsos, num_dlas_samples)
        self.sample_log_likelihoods_lls = self.processed_file['sample_log_likelihoods_dla'][()].T
        self.sample_log_likelihoods_lls = self.sample_log_likelihoods_lls[~self.nan_inds, :]

        # shape (num_dlas, num_qsos); could be derived from sample_log_likelihoods_dla
        # load with shape (num_qsos, num_dlas_samples)
        self.log_likelihoods_dla = self.processed_file['log_likelihoods_dla'][()].T
        self.log_likelihoods_dla = self.log_likelihoods_dla[~self.nan_inds, :]

        # make sure the occam's razor normalization is correct
        assert np.abs( self.log_likelihoods_dla[-1, 0] - 
            lognanmeanexp( self.sample_log_likelihoods_dla[-1, :, 0])
            ) / self.log_likelihoods_dla[-1, 0] < 1e-2

        # a series of matrices with shape (1, num_qsos)
        # load with shape (num_qsos, )
        self.log_likelihoods_lls    = self.processed_file['log_likelihoods_lls'][0, :]
        self.log_likelihoods_lls    = self.log_likelihoods_lls[~self.nan_inds]
        self.log_likelihoods_no_dla = self.processed_file['log_likelihoods_no_dla'][0, :]
        self.log_likelihoods_no_dla = self.log_likelihoods_no_dla[~self.nan_inds]

        self.log_priors_lls         = self.processed_file['log_priors_lls'][0, :]
        self.log_priors_no_dla      = self.processed_file['log_priors_no_dla'][0, :]

        # the base inds, can be used to find out the parameters of a given sample at different
        # DLA models; in shape (num_dlas - 1, num_dla_samples, num_qsos)
        # load with shape (num_qsos, num_dla_samples, num_dlas - 1)
        # the first layer of the matrix is for DLA(2)
        self.base_sample_inds = self.processed_file['base_sample_inds'][()].T
        self.base_sample_inds = self.base_sample_inds[~self.nan_inds, :, :]

        # loading sample file; load with shape (num_dla_samples, )
        self.sample_filehandle = h5py.File(sample_file, 'r')
        self.log_nhi_samples   = self.sample_filehandle['log_nhi_samples'][:, 0]

        # take adventage of large memory usage, load every single log_nhi samples
        # into a huge matrix with the same size as the sample_log_likelihoods_dla
        # in shape (num_qsos, num_dla_samples, num_dlas - 1)
        self.multi_log_nhi_samples = self.log_nhi_samples[self.base_sample_inds]

        log_nhi_dla1 = np.repeat(
            np.arange(num_dla_samples)[None, :], num_qsos, axis=0 )[:, :, None]

        # concat into shape (num_qsos, num_dla_samples, num_dlas)
        self.multi_log_nhi_samples = np.concatenate(
            (log_nhi_dla1, self.multi_log_nhi_samples), axis=2 )

        # load subDLA logNHI samples; load with shape (num_dla_samples, )
        self.subdla_sample_filehandle = h5py.File(sub_dla_sample_file, 'r')
        self.lls_log_nhi_samples      = self.subdla_sample_filehandle['lls_log_nhi_samples'][:, 0]

        # get the initial log_poetriors
        self.evaluate_log_posteriors()

    @property
    def _sample_log_likelihoods_dla(self):
        '''
        (num_qsos, num_dla_samples, num_dlas)
        '''
        return self.sample_log_likelihoods_dla

    def evaluate_log_posteriors(self) -> None:
        '''
        Evaluate the log posteriors of having DLAs or not using log_priors and log_likelihoods
        
        log_posteriors = log_priors + log_likelihoods
        '''
        assert self.log_priors_dla.shape == self.log_likelihoods_dla
        self.log_posteriors_dla = self.log_priors_dla + self.log_likelihoods_dla

        assert self.log_priors_no_dla.shape == self.log_likelihoods_no_dla.shape
        self.log_posteriors_no_dla = self.log_priors_no_dla + self.log_likelihoods_no_dla

        # prevent users think subDLA posteriors do not included in the noDLA posteriors
        assert self.log_likelihoods_lls.shape == self.log_priors_lls.shape 
        log_posteriors_lls = self.log_priors_lls + self.log_likelihoods_lls

        # combine subDLAs to noDLA model posteriors
        self.log_posteriors_no_dla = logsumexp(
            (log_posteriors_lls, self.log_likelihoods_no_dla), axis=0)
        
        # combine multi-dim log posteriors dla to 1-dim array
        self.log_posteriors_dla = logsumexp(
            self.log_posteriors_no_dla, axis=1)
        
        assert self.log_posteriors_no_dla.shape == self.log_posteriors_dla.shape

    def update_log_posteriors(self, log_nhi_cut: float = 20.) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Recalculate model posteriors based on a given sub-DLA upper limit, log_nhi_cut.

        if cut < 20:
            P(DLA) = ( P(DLA | logNHI > cut) + P(subDLA | logNHI > cut ) ) /
                     ( P(DLA) + P(subDLA) + P(no DLA) )
        else:
            P(DLA) =   P(DLA | logNHI > cut) /
                     ( P(DLA) + P(subDLA) + P(no DLA) )

        :param log_nhi_cut: the minimum cutoff for being a DLA
        :return: (log_posteriors_dla, log_posteriors_no_dla)
        '''
        assert (log_nhi_cut < 23) & (log_nhi_cut > 19.5)
        # select all dlas inds; shape (num_qsos, num_dla_samples, num_dlas)
        ind_dla     = (self.multi_log_nhi_samples > log_nhi_cut)

        # select subDLAs to be DLAs; shape: (num_dla_samples, )
        ind_lls_dla = (self.lls_log_nhi_samples   > log_nhi_cut)

        num_qsos = self.log_likelihoods_dla.shape[0]
        log_posteriors_dla    = np.empty((num_qsos, ), dtype=float)
        log_posteriors_no_dla = np.empty((num_qsos, ), dtype=float)

        for quasar_ind in range(num_qsos):
            # P(DLA) = P(DLA | logNHI > cut) + P(subDLA | logNHI > cut )
            this_sample_log_posteriors_dla = self.sample_log_likelihoods_dla[quasar_ind, :, :]
            this_sample_log_posteriors_dla += self.log_priors_dla[quasar_ind, None, :]

            this_log_posteriors_dla = lognanmeanexp( 
                this_sample_log_posteriors_dla[ind_dla[quasar_ind, :, :]]
                )

            # make a copy of sample log posteriors lls first and then apply the indexing
            # if do it in a reverse way, then empty index will result in log priors lls,
            # but we actually want no posteriors if index slicing is empty.
            this_sample_log_posteriors_lls = self.sample_log_likelihoods_lls[quasar_ind, :]
            this_sample_log_posteriors_lls += self.log_priors_lls[quasar_ind]

            # prevent zero-size array
            if np.any(ind_lls_dla[quasar_ind, :]) == False:
                this_log_posteriors_lls_dla = -np.inf # prob = 0
            else: 
                this_log_posteriors_lls_dla = lognanmeanexp(
                    this_sample_log_posteriors_lls[ind_lls_dla[quasar_ind, :]]
                    )

            # log ( P(DLA | NHI > cut) + P(subDLA | NHI > cut) )
            log_posteriors_dla[quasar_ind] = logsumexp(
                ( this_log_posteriors_dla, this_log_posteriors_lls_dla ), axis=0 )

            # Now P(no DLA) = P(no DLA) + P(subDLA | logNHI < cut) + P(DLA | logNHI < cut)
            # log P(DLA | logNHI < cut)
            if np.any(~ind_dla[quasar_ind, :, :]) == False:
                this_log_posteriors_dla_no_dla = -np.inf
            else:
                this_log_posteriors_dla_no_dla = lognanmeanexp(
                    this_sample_log_posteriors_dla[~ind_dla[quasar_ind, :, :]]
                    )

            # log P(subDLA | logNHI < cut )
            this_log_posteriors_lls = lognanmeanexp(
                this_sample_log_posteriors_lls[~ind_lls_dla[quasar_ind, :]]
                )

            # log(  P(no DLA) + P(subDLA | logNHI < cut) + P(DLA | logNHI < cut) )
            log_posteriors_no_dla[quasar_ind] = logsumexp( 
                (self.log_posteriors_no_dla[quasar_ind], this_log_posteriors_lls,
                this_log_posteriors_dla_no_dla), axis=0 )

            del this_sample_log_posteriors_dla, this_sample_log_posteriors_lls
            del this_log_posteriors_dla, this_log_posteriors_dla_no_dla
            del this_log_posteriors_lls, this_log_posteriors_lls_dla

        return log_posteriors_dla, log_posteriors_no_dla

    # log_likelihoods_dla(quasar_ind, num_dlas) = ...
    #     max_log_likelihood + log(nanmean(sample_probabilities)) ...
    #     - log( num_dla_samples ) * (num_dlas - 1); % occam's razor
