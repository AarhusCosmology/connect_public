#!/bin/bash 
output_dir="chains/connect_${1}_data"
output_file="chains/connect_out/${1}.out"
MPARGS=" -p ${2}"
MPARGS+=" -o $output_dir"
MPARGS+=" -c montepython_public/covmat/base2018TTTEEE_lite.covmat"
MPARGS+=" -N 5000000000"
MPARGS+=" --conf connect.conf"
MPARGS+=" -j fast -f 2.1 --silent"
MPARGS+=" --update 1000"
MPARGS+=" -T 5.0"
rm -rf $output_dir
source activate MontePythonEnvironment
source planck2018/code/plc_3.0/plc-3.01/bin/clik_profile.sh

mp_tol=$3
node=$4

if ! [ -z $node ]
then
    MPI_ARGS="--oversubscribe -np 4 --host $node"
else
    MPI_ARGS="--oversubscribe -np 4"
fi

mpirun $MPI_ARGS python montepython_public/montepython/MontePython.py run $MPARGS > $output_file & pid=$!

job_running=true

while $job_running
do
    sleep 5s
    line0=$( grep -Fn 'Scanning file' $output_file | tail -n 1 | tr ":" "\n" | head -n 1)
    line1=$( grep -Fn 'R-1 is' $output_file | tail -n 1 | tr ":" "\n" | head -n 1)
    line2=$(( $( grep -Fn 'covariance matrix' $output_file | tail -n 1 | tr ":" "\n" | head -n 1) - 1 ))
    if [ -z "$line1" ]
    then
        line1=0
    fi
    if [ -z "$line2" ]
    then
        line2=0
    fi
    
    if [ $line2 -gt $line1 ] && ! [[ $(sed -n "${line0},$$p" $output_file) == *"Removed everything: chain not converged"* ]]
    then
        kill_job=true
        for (( l=$line1; l<=$line2; l++ ))
        do
            GR=$(sed -n ${l}p $output_file | tr "\t" "\n" | head -n 1)
            GR=${GR/ -> R-1 is /}
            if (( $(echo "$GR > $mp_tol" | bc -l) ))
            then
                kill_job=false
            fi
        done
        if $kill_job
        then
            kill $pid
            job_running=false
        fi
    fi
done
conda deactivate
