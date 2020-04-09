#!/bin/bash

# download_spectra_dr12q.sh: downloads DR14Q spectra from SDSS

base_directory='..'
pushd $base_directory/dr14q/spectra

rsync --info=progress2 -h --no-motd --files-from=file_list rsync://data.sdss.org/dr16/eboss/spectro/redux/ . 2> /dev/null
