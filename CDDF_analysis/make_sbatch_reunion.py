'''
Combine pieces of .mat files into a single file.
Assume .mat is -v7.3 so HDF5 reader would work
'''
from typing import List

from .set_parameters import *
from .sbatch_reunion import mat_combine
import os, glob
import re

def combine_custom(out_filename: str, release: str, chunk_size: int = 5000,
        maxshape_size: int =162861, test: bool=False, subfix: str = "", small_file: bool = False) -> List:
    '''
    combine pieces of .mat files into a .mat file using h5py,
    with a given out_filename and release.

    Parameters:
    ----
    `out_filename` (str) : e.g., processed_qsos_multi_lls_dr12q.mat
    `release` (str)      : e.g., dr12q
    `subfix` (str)       : e.g., processed_qsos_multi_lls_dr12q_{num1}-{num2}{subfix}.mat,
        num1-num2, the qso_ind we in the piece of processed file.
    `small_file` (bool)  : if you do not want to save sampling results, e.g., sample_log_likelihoods,
        sample_log_posteriors

    Returns:
    ----
    List, list of filenames, and will create a out_filename{subfix}.mat in the execution directory
    '''
    out_prefix = out_filename.split('.')[0]

    # a list of processed .mat files
    processed_files = [
        "{}_{}-{}{}.mat".format(out_prefix, i + 1, i + chunk_size + 1, subfix) 
        if (i + chunk_size) <= maxshape_size
        else 
        "{}_{}-{}{}.mat".format(out_prefix, i + 1, maxshape_size + 1, subfix)        
        for i in range(0, maxshape_size, chunk_size)
    ]

    # add processed directory
    processed_files = [
        os.path.join( processed_directory(release), f )
        for f in processed_files
    ]

    if test:
        print(processed_files)
        assert all([os.path.exists(f) for f in processed_files])
        print("[Info] All pieces of files exist.")
    else:
        # recombine all pieces of files 
        mat_combine(processed_files, out_filename, chunk_size, maxshape_size, small_file=small_file)

    return processed_files

def combine_adaptive(out_filename: str, release: str, maxshape_size: int = 162861,
        test: bool = False, subfix: str = "", start_quasar_ind: int = 1, small_file: bool = False) -> List:
    '''
    find all file names with <out_prefix_[0-9]+_[0-9]+.mat>

    use the second match to search the next filename
    '''
    out_prefix = out_filename.split('.')[0]

    pattern_search = lambda start, string : (re.findall(
        '{}_{}-([0-9]+){}.mat'.format(out_prefix, start, subfix), string))
    
    # loop through all filename with the same prefix
    all_filenames = glob.glob('{}*'.format(
        os.path.join(processed_directory(release), out_prefix) ))

    end_quasar_ind   = 0

    counter = 0

    processed_files  = []

    while end_quasar_ind != maxshape_size:
        end_quasar_ind, filename = next(
            regex_prefix(pattern_search, all_filenames, start_quasar_ind))
        # append this filename
        processed_files.append(filename)

        # update strart number
        start_quasar_ind = end_quasar_ind

        # get the first chuck_size
        if counter == 0:
            chunk_size = start_quasar_ind

        counter += 1
        if counter > len(all_filenames):
            return None

    if test:
        assert all([os.path.exists(f) for f in processed_files])
        print(processed_files)
        print("[Info] All pieces of files exist.")
    else:
        # recombine all pieces of files 
        mat_combine(processed_files, out_filename, chunk_size, maxshape_size, small_file=small_file)

    return processed_files

def regex_prefix(pattern_search, all_filenames, start_quasar_ind):
    '''
    A generator to find the matched string in the list
    '''
    for filename in all_filenames:
        end_quasar_ind = pattern_search(start_quasar_ind, filename)
        if len(end_quasar_ind) > 0:
            yield int(end_quasar_ind[0]), filename
