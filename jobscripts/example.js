#!/bin/bash
#SBATCH --job-name=create_data
#SBATCH --partition=q24,q28,q36,q40,q48
#SBATCH --mem-per-cpu=2g
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=test.out
echo "========= Job started at `date` =========="

# activate proper environment if needed
source activate ConnectEnvironment
module load gcc openmpi

# source planck data (load path from connect.conf)
clik_line=$(grep -hr "clik" mcmc_plugin/config_files/connect.conf)
path_split=(${clik_line//= / })
# if zsh is used, set var to clik_profile.zsh, else set it to clik_profile.sh
if [ -n "$ZSH_VERSION" ]; then
    var="clik_profile.zsh"
else
    var="clik_profile.sh"
fi
path="$(echo ${path_split[1]} | sed "s/'//g")bin/$var"
source $path

python connect.py create input/example_class.param

echo "========= Job finished at `date` =========="
