# sort out the test_ind from two different catalogs

import h5py
import numpy as np
import hdf5storage

catalog1 = "data/dr12q/processed/zqso_only_catalog.mat"
catalog2 = "data/dr12q/processed/catalog_jfaub.mat"

catalog1 = h5py.File(catalog1, 'r')
catalog2 = h5py.File(catalog2, 'r')

thing_ids = catalog1['thing_ids'][0, :] # should be the same

filter_flags1 = catalog1['filter_flags'][0, :]
filter_flags2 = catalog2['filter_flags'][0, :]

# get the test ind from filter flags
test_ind1 = filter_flags1 == 0
test_ind2 = filter_flags2 == 0

real_index1 = np.where(test_ind1)[0]
real_index2 = np.where(test_ind2)[0]

test_mf2jf_ind = np.isin(real_index1, real_index2)
test_jf2mf_ind = np.isin(real_index2, real_index1)

# acquire the missing index need to be run
need_to_run = real_index2[~test_jf2mf_ind]
assert np.all(need_to_run)

test_ind_need_to_run = np.zeros(len(test_ind1), dtype="bool")
test_ind_need_to_run[need_to_run] = True

print("{} need to re-run".format(len(need_to_run)))

# store into matlab compatible file
test_ind_file = {
    "test_ind"  : test_ind_need_to_run[:, None],
    "thing_ids" : thing_ids[test_ind_need_to_run].astype(np.int)[:, None],
    "filter_flagsMF" : filter_flags1[:, None],
    "filter_flagsJF" : filter_flags2[:, None]
    }

hdf5storage.write(test_ind_file, '.', 'test_rerun_ind.mat', matlab_compatible=True)

# store missing thing_ids into a txt file
with open("thingIDs_need_to_run.txt", "w") as f:
    f.write("# thingID\n")
    for thingid in thing_ids:
        f.write("{}\n".format(int(thingid)))

# select the later half of the test_ind for Jacob's merge
later_half1 = real_index1[80000:]
assert later_half1.shape[0] == 82861
later_half2 = real_index2[80000:]

# map my index to Jacob's index
mf2jf_ind = np.isin(later_half1, later_half2)
jf2mf_ind = np.isin(later_half2, later_half1)

thing_ids_mf = thing_ids[later_half1]
thing_ids_jf = thing_ids[later_half2]

assert np.all( thing_ids_jf[jf2mf_ind] == thing_ids_mf[mf2jf_ind] )

mf2jf_ind_file = {
    'mf2jf_ind' : mf2jf_ind[:, None],
    'jf2mf_ind' : jf2mf_ind[:, None], # processed_jf[jf2mf_ind] = processed_mf[mf2jf_ind]
}

hdf5storage.write(mf2jf_ind_file, '.', 'mf2jf_ind_file.mat', matlab_compatible=True)
