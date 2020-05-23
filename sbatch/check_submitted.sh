# check if we have already got the processed file,
# if not, re-submit the job

base_directory="data/dr12q/processed/"

processed_prefix="processed_qsos_zqsos_sbird_dr12q-100_"
processed_filename="_norm_1176-1256.mat"

quasar_start_ind=150001
quasar_end_ind=160000
chunk=50

num_quasars=$(((quasar_end_ind - quasar_start_ind + 1)/chunk))

for ((i=1; i <= $num_quasars; i++ ));
do
    qso_ind_start=$((quasar_start_ind + (i - 1) * chunk))
    qso_ind_end=$((quasar_start_ind +  i      * chunk))   # my saving condition in process_qsos.m has +1 diff
    echo "checking processed file ..."

    file="${base_directory}${processed_prefix}${qso_ind_start}-${qso_ind_end}${processed_filename}"

    # check if the processed file is in the directory
    if [ -f "$file" ]
    then
        echo "$file found!"
    else
        echo "$file missing, re-submit"
        # bash submit_gp_zqsos2.sh $qso_ind_start $qso_ind_end `pwd`

        # biocluster would reject my MATLAB parallel jobs if submitting a sequence of jobs too fast
        sleep 5
    fi
done
