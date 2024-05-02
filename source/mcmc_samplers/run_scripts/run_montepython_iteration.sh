#!/bin/bash 
output_dir=$1
output_file="${1}/montepython.log"
MPARGS=" -p ${2}"
MPARGS+=" -o $output_dir"
MPARGS+=" -c covmat/base2018TTTEEE_lite.covmat"
MPARGS+=" -N 1000000"
MPARGS+=" --conf ${3}"
MPARGS+=" -j fast -f 2.1 --silent"
MPARGS+=" --update 1000"
MPARGS+=" -T ${4}"
mkdir -p $output_dir

mcmc_tol=$5
node=$6

if ! [ $node == "None" ]
then
    MPI_ARGS=" -np 4 --host $node"
else
    MPI_ARGS=" -np 4"
fi

mpirun $MPI_ARGS python montepython/MontePython.py run $MPARGS &> $output_file & pid=$!

job_running=true

while [ $job_running = true ] && [ $(ps -p $pid | wc -l) -gt 1 ]
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
            bool1=$(echo "$GR > $mcmc_tol" | bc -l 2> /dev/null)
            bool2=$(echo "$GR==0" | bc -l 2> /dev/null)
            a=$(echo "sqrt($bool1 + $bool2)" | bc -l)
            bool=${a%.*}
	    if [ -z $bool ]
	    then
		bool="0"
	    fi
            if (( $bool ))
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

sleep 1s # Ensures that buffer has time to be emptied upon termination of Monte Python
