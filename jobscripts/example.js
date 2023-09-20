#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --partition=q24,q28,q36,q40
#SBATCH --mem=40g
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=test4.out
echo "========= Job started at `date` =========="

# activate proper environment if needed
#ml load anaconda3/5.0.1
source ~/.bashrc
ml load anaconda3/5.0.1 gcc openmpi
source activate ConnectEnvironment
cd $SLURM_SUBMIT_DIR

# source planck data (load path from connect.conf)
clik_line=$(grep -hr "clik" mcmc_plugin/connect.conf)
path_split=(${clik_line//= / })
path="$(echo ${path_split[1]} | sed "s/'//g")bin/clik_profile.sh"
source $path

python connect.py create input/example.param

echo "========= Job finished at `date` =========="
