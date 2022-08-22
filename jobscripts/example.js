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
# source planck data
python connect.py create input/example.param
echo "========= Job finished at `date` =========="
