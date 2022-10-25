#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --partition=q24,q28,q36,q40,q48
#SBATCH --mem=40g
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=test.out
echo "========= Job started at `date` =========="

# activate proper environment if needed
source activate ConnectEnvironment

# source planck data (load path from connect.conf)
clik_line=$(grep -hr "clik" mp_plugin/connect.conf)
path_split=(${clik_line//= / })
path="$(echo ${path_split[1]} | sed "s/'//g")bin/clik_profile.sh"
source $path

python connect.py create input/example.param

echo "========= Job finished at `date` =========="
