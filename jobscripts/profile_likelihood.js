#!/bin/bash
#SBATCH --job-name=profile_triangle_plot
#SBATCH --partition=q48,q40,q36,q28
#SBATCH --mem-per-cpu=8g
#SBATCH --ntasks=500
#SBATCH --cpus-per-task=1
#SBATCH --time=04:50:00
#SBATCH --output=profile_likelihood.out
echo "========= Job started at `date` =========="

source activate ConnectEnvironment
ml load gcc openmpi

mpirun -np 500 python connect.py profile -m trained_models/<model_name> -c <chain_folder> -o <output_name>
# specify additional options (see 'python connect.py profile -h')
echo "========= Job finished at `date` =========="
