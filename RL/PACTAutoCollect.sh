#!/bin/bash

# Source conda env
source activate PACT

exppath=./exp/result
mkdir -p ${exppath}

result_path=./PACTData
mkdir -p ${result_path}

declare -a ENVS=("finger_turn_hard" "cheetah_run" "humanoid_run" "quadruped_run")
declare -a NAMES=("rlfthrun_100000" "rlchtrun_100000" "rlhumrun_100000" "rlqdrun_100000")

for index in "${!ENVS[@]}";do
    name="${NAMES[index]}"
    env="${ENVS[index]}"
    for trial in {1..5}; do
        echo Starting: "$name"_"$trial"
        python3 ./train.py env=${env} > out_"$trial".txt 2>&1
        sleep 2
        mv ${exppath}/PACT.csv ${result_path}/"$name"_"$trial"_PACT.csv
        mv out_"$trial".txt ${result_path}/"$name"_"$trial"_out.txt           
        sleep 2
    done
done

# Deactivate conda env
conda deactivate

echo "Finished: Exiting!"
