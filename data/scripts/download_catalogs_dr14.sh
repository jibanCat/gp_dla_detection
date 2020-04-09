#!/bin/bash

# downlaod_catalogs.sh: downloads SDSS DR9Q, DR10Q, and DR12Q
# catalogs, as well as corresponding previously compiled DLA catalogs

base_directory='..'
pushd $base_directory

# DR14Q
directory='dr14q'

mkdir -p $directory/spectra $directory/processed $directory/distfiles
pushd $directory/distfiles
filename='DR14Q_v4_4.fits'
wget https://data.sdss.org/sas/dr16/eboss/qso/DR14Q/$filename -O $filename
popd
