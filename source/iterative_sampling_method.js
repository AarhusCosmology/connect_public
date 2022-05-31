#!/bin/bash
#SBATCH --job-name=iter_gpu
#SBATCH --partition=q48
#SBATCH --mem=20G
#SBATCH --time=10:00:00
echo "========= Job started at `date` =========="
ml load anaconda3/2021.05
CONNECT_PATH=$1
Parameter_file=$2
passphrase=$3
training_resource=$4
cd $CONNECT_PATH

echo "CPUs allocated"

if [ $training_resource = "gpu" ]
then
    jobid=$(sbatch --job-name=training -p qgpuo --gres=gpu:2 --exclude=s27n[01,02] --time=10:00:00 --mem=6g --wrap="while true; do sleep 10000s; done" | tr -dc '0-9')

    status=$(squeue --job $jobid 2> /dev/null | tail -n 1 | tr " " "\n" | sed -r '/^\s*$/d' | sed -n 5p)
    while ! [ $status = "R" ]
    do
	sleep 5s
	status=$(squeue --job $jobid 2> /dev/null | tail -n 1 | tr " " "\n" | sed -r '/^\s*$/d' | sed -n 5p)
    done

    echo "GPUs allocated"

    gpu_node=$(squeue --job $jobid 2> /dev/null | tail -n 1 | tr " " "\n" | sed -r '/^\s*$/d' | sed -n 8p)

fi

#out=$(scontrol show job -d $SLURM_JOB_ID)
mp_node=$(python -c "from source.misc_functions import get_node_with_most_cpus as gnwmc; print(gnwmc())")



names=( "initial_model" "montepython_path" "jobname" "save_name" "overwrite_model" "N_initial" "N_cpu" "N_max_lines" "batchsize" "epochs" "resume_iterations" "keep_first_iteration")
names_str=$(echo "${names[@]}" | tr ' ' '\n')

values=( $(python -c "from source.misc_functions import get_param_attributes as gpa; print(gpa('$Parameter_file',\"\"\"${names_str}\"\"\"))") )

for i in "${!names[@]}"
do 
    eval "${names[i]}=${values[i]}"
done

montepython_paramfile=$(python -c "from source.misc_functions import create_montepython_param as cmp; print(cmp('$Parameter_file','$montepython_path'))")



if [ $keep_first_iteration = "True" ]
then
    keep_idx="0"
else
    keep_idx="1"
fi


if [ $resume_iterations = "False" ]
then
    rm -rf data/$jobname
    mkdir data/$jobname
    cp $Parameter_file data/$jobname/log_connect.param
    Parameter_file="${CONNECT_PATH}/data/${jobname}/log_connect.param"
    if [ $(awk "/^jobname/{print}" data/$jobname/log_connect.param | wc -l) == "0" ]
    then
	echo "jobname = '${jobname}'" >> data/$jobname/log_connect.param
    fi
    mkdir data/$jobname/compare_iterations
    python -c "from source.misc_functions import Gelman_Rubin_log as grl; grl('$Parameter_file',status='initial')"
    i="1"
    if [ $initial_model = "None" ]
    then
	echo "No initial model chosen - creating from scratch..."
	python -c "from source.misc_functions import create_lhs as cl; cl('$Parameter_file')"
	python -c "from source.misc_functions import create_output_folders as cof; cof('$Parameter_file')"
	
	echo "Calculating ${N_initial} CLASS models"
	source activate mpienv
	export OMP_NUM_THREADS=1
	srun -n `expr $N_cpu + 1` python source/calc_models_mpi.py data/$jobname/log_connect.param $CONNECT_PATH lhs > err_calc_$jobname.txt
	conda deactivate
    fi
else
    Parameter_file="${CONNECT_PATH}/data/${jobname}/log_connect.param"
    path="data/${jobname}/number_"
    last_iter=$(ls -d $path* | tr "$path" " " | xargs | tr ' ' '\n' | sort -n | tail -n 1)
    if ! [ $last_iter -gt "0" ]
    then
	echo "Nothing to resume - exiting sampling"
	scancel $jobid
	exit 1
    fi
    if [ -f "${path}${last_iter}/model_params.txt" ]
    then
	last_iter=`expr $last_iter + 1`
    else
	rm -rf $path$last_iter
    fi
    i=$last_iter
    j=`expr $i - 1`
    N_initial=`expr $(wc -l data/$jobname/number_$j/model_params.txt | tr " " "\n" | head -n 1) - 1`
    echo "Restarting from iteration ${last_iter}"
fi

if [ $resume_iterations != "False" ] || [ $initial_model = "None" ]
then
    name_base="bs${batchsize}_e${epochs}"

    echo "Training neural network"
    if [ $initial_model = "None" ] && [ $resume_iterations = "False" ]
    then
	j="0"
	N_ini=$N_initial
    else
	j=`expr $i - 1`
	N_ini="0"
    fi

    if [ $training_resource = "gpu" ]
    then
	initial_model=$($CONNECT_PATH/source/shell_scripts/train_iteration_gpu.sh $Parameter_file $jobname $j $N_ini $save_name $name_base $gpu_node $passphrase $CONNECT_PATH)
    else
	initial_model=$($CONNECT_PATH/source/shell_scripts/train_iteration_cpu.sh $Parameter_file $jobname $j $N_ini $save_name $name_base none $passphrase $CONNECT_PATH)
    fi

fi





model=$initial_model
echo "Initial model is ${model}"


while true
do
    cd $montepython_path
    echo "Beginning iteration no. ${i}"
    model_str="data.cosmo_arguments['connect_model']"
    lines=$(grep -F $model_str $montepython_paramfile)
    line=$(while IFS= read -r line; do if [[ "$line" == "${model_str}"* ]]; then echo $line; fi; done <<< "$lines" | head -n 1)
    line_number=$(if ! [ -z "$line" ]; then grep -Fn "$line" $montepython_paramfile | tr ":" "\n" | head -n 1; else echo "-1"; fi)
    if ! [ $line_number == "-1" ]
    then
	sed -i "${line_number}s/.*/${model_str} = '${model}'/" $montepython_paramfile
    else
	echo "${model_str} = '${model}'" >> $montepython_paramfile
    fi

    echo "Running MCMC sampling no. ${i}..."
    $CONNECT_PATH/source/shell_scripts/run_montepython_iteration.sh $jobname $montepython_paramfile $mp_node 2> out_error_$jobname.txt
    echo "MCMC sampling stopped since R-1 less than 0.01 has been reached"

    echo "Number of accepted steps: $(cat chains/connect_${jobname}_data/*.txt | wc -l)"
    cp -r chains/connect_${jobname}_data $CONNECT_PATH/data/$jobname/number_$i
    cd $CONNECT_PATH

    if [ $i = "1" ] && [ $keep_first_iteration != "True" ]
    then
	N_keep="5000"
    else
	N_keep=$N_max_lines
    fi
    N_keep=$(python -c "from source.misc_functions import filter_chains as fc; print(fc('$Parameter_file','data/${jobname}/number_${i}',$N_keep,$i))")
    echo "Keeping only last ${N_keep} of the accepted Markovian steps"

    if [ $i -gt "1" ]
    then
	echo "Comparing latest iterations..."
	chain1="data/${jobname}/compare_iterations/chain__$(echo `expr $i - 1`).txt"
	chain2="data/${jobname}/compare_iterations/chain__${i}.txt"
	output=$(python2 ${montepython_path}/montepython_public/montepython/MontePython.py info $chain1 $chain2 --noplot --minimal)
	out_GR=$(python -c "from source.misc_functions import Gelman_Rubin_log as grl; print(grl('$Parameter_file',status='$i',output=\"\"\"${output}\"\"\"))")
	echo "$out_GR" | head -n -1
	kill_iteration=$(echo "$out_GR" | tail -n 1)
	if [ $kill_iteration = "True" ]
	then
	    echo "iteration ${i} will be the last since convergence in data has been reached"
	    #rm -rf data/$jobname/number_$i
	    #break
	fi
    fi

    if [ $i -gt `expr $keep_idx + 1` ]
    then
	model_params="data/${jobname}/number_$(echo `expr $i - 1`)/model_params.txt"
	N_accepted=$(python -c "from source.misc_functions import discard_oversampled_points as dop; print(dop('$model_params','$Parameter_file',$i))")
	echo "Accepted $N_accepted points out of $N_keep"
    else
	N_accepted=$N_keep
    fi
    


    python -c "from source.misc_functions import create_output_folders as cof; cof('$Parameter_file',$i)"
    echo "Calculating ${N_accepted} CLASS models"
    source activate mpienv
    export OMP_NUM_THREADS=1
    srun -n `expr $N_cpu + 1` python source/calc_models_mpi.py data/$jobname/log_connect.param $CONNECT_PATH iterative > err_calc_${jobname}_${i}.txt #/dev/null
    conda deactivate

    python -c "from source.misc_functions import join_data_files as jdf; jdf('$Parameter_file')"

    if [ $i -gt `expr $keep_idx + 1` ]
    then
	python -c "from source.misc_functions import combine_iterations_data as cid; cid('$Parameter_file',$i)"
	echo "Copied data from data/${jobname}/number_$(echo `expr $i - 1`) into data/${jobname}/number_${i}"
    fi

    name_base="bs${batchsize}_e${epochs}"
    echo "Training neural network"
    if [ $training_resource = "gpu" ]
    then
	model=$($CONNECT_PATH/source/shell_scripts/train_iteration_gpu.sh $Parameter_file $jobname $i "0" $save_name $name_base $gpu_node $passphrase $CONNECT_PATH)
    else
	model=$($CONNECT_PATH/source/shell_scripts/train_iteration_cpu.sh $Parameter_file $jobname $i "0" $save_name $name_base none $passphrase $CONNECT_PATH)
    fi

    i=`expr $i + 1`
    if ! [ -z $kill_iteration ]
    then
	if [ $kill_iteration = "True" ]
	then
	    echo "Final model is ${model}"
	    break
	else
	    echo "New model is ${model}"
	fi
    fi
done

scancel $jobid

echo "========= Job finished at `date` =========="
