#!/bin/bash
#./submit_gp_zqsos2.sh 1 1000 `pwd`


sbatch <<EOT
#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mem=128G
#SBATCH --job-name=gp_zqsos_sbird_$1-$2
#SBATCH -p short
#SBATCH --output="gp_zqsos_sbird_$1-$2.log"

date
cd $3

echo "----"

pwd

# load matlab module
module load matlab

# run matlab script
matlab -nodesktop -nosplash -r "num_cores = 32; num_threads = 32; qso_ind = $1:$2; process_zestimate; exit;"

hostname
EOT
