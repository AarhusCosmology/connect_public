#!/bin/bash

if [ -z $SLURM_NPROCS ]
then
     N_tasks=$(nproc)
     N_cpus_per_task="1"
else
     N_tasks=$SLURM_NPROCS
     N_cpus_per_task=$SLURM_CPUS_PER_TASK
fi

echo "${N_tasks},${N_cpus_per_task}"
