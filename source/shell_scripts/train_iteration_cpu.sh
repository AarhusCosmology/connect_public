#!/bin/bash

Parameter_file=$1
jobname=$2
i=$3
N_initial=$4
save_name=$5
name_base=$6
cpu_node=$7
passphrase=$8
CONNECT_PATH=$9

train_args=$Parameter_file
while true
do
    python connect.py train $train_args 1> training_iter_${i}_${jobname}.log 2> /dev/null
    
    if grep -Fq "nan" $CONNECT_PATH/training_iter_${i}_${jobname}.log
    then
	train_args="${train_args} 100"
    else
	break
    fi
done


if ! [ $save_name == "None" ]
then
    initial_model=$save_name
else
    if [ $N_initial -gt "0" ]
    then
	N_data=`expr $(wc -l data/$jobname/N-$N_initial/model_params.txt | tr ' ' '\n' | head -n 1) - 1`
    else
	N_data=`expr $(wc -l data/$jobname/number_$i/model_params.txt | tr ' ' '\n' | head -n 1) - 1`
    fi
    initial_model="${jobname}_N${N_data}_${name_base}"
fi
if ! [[ $overwrite_model == "True" ]]
then
    prefix="trained_models/"
    if [ -d "$prefix$initial_model" ]
    then
        num="1"
        name_not_found="True"
        while [ $name_not_found = "True" ]
        do
            if [ -d "${prefix}${initial_model}_${num}" ]
            then
                num=`expr $num + 1`
            else
                num=`expr $num - 1`
                name_not_found="False"
                if ! [ $num = "0" ]
                then
                    initial_model="${initial_model}_${num}"
                fi
            fi
        done
    fi
fi

echo $initial_model
