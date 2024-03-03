#!/bin/bash

# Load modules
module load gcc/11.1.0
module load anaconda/3/2019.03
module load cuda/11.8

# Source conda env
source activate PACT

result_path=./PACTData
mkdir -p ${result_path}

declare -a MODELS=("vgg" "resnet" "densenet" "mobilenet")
declare -a NAMES=("vgg_train_200" "resnet_train_200" "densenet_train_50" "mobilenet_train_300")
declare -a EPOCHS=(200 200 50 300)


for index in "${!MODELS[@]}";do
    name="${NAMES[index]}"
    model="${MODELS[index]}"
    epoch=${EPOCHS[index]}
    for trial in {1..5}; do
        echo Starting: "$name"_"$trial"
        python3 main.py --epochs=${epoch} --model=${model} --res=${result_path} > out_"$trial".txt 2>&1
        sleep 2
        mv ${result_path}/PACT.csv ${result_path}/"$name"_"$trial"_PACT.csv
        mv out_"$trial".txt ${result_path}/"$name"_"$trial"_out.txt          
        sleep 2
    done
done

# Deactivate conda env
conda deactivate

echo "Finished: Exiting!"