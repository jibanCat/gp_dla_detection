# batch submit zqsos2- zestimation with Lya forest script
# note that it is around 50 seconds per spec,
# so split into 100 spec chucks is around ~1.5 hrs,
# short node has maximum 2 hrs and we want to use short node.

quasar_start_ind=150001
quasar_end_ind=160000
chunk=100

num_quasars=$(((quasar_end_ind - quasar_start_ind + 1)/chunk))

for ((i=1; i <= $num_quasars; i++ ));
do
    qso_ind_start=$((quasar_start_ind + (i - 1) * chunk))
    qso_ind_end=$((quasar_start_ind +  i      * chunk - 1))
    echo "submitting GP job: from $qso_ind_start to $qso_ind_end; $i / $num_quasars"

    bash submit_gp_zqsos2.sh $qso_ind_start $qso_ind_end `pwd`
done
