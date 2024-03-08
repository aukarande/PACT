#!/bin/bash

# Source conda env
source activate PACT

dataset_path=./data/sst2_small_dataset.tsv
cross_validation_folder=./data/sst2_cvp
k_fold=5
# End here

#Spliting the dataset into defined K-folds 
#Comment this section if you have your dataset splitted into K-Folds
python split.py \
 -dataset_path ${dataset_path} \
 -cross_validation_path ${cross_validation_folder} \
 -k_fold ${k_fold}

result_path=./PACTData
mkdir -p ${result_path}

declare -a MODELS=("prajjwal1/bert-small" "prajjwal1/bert-medium" "bert-base-uncased" "bert-large-uncased" "distilbert-base-uncased")
declare -a NAMES=("berts_train_100" "bertm_train_65" "bertbu_train_50" "bertlu_train_25" "dbert_train_75")
declare -a EPOCHS=(100 65 50 25 75)


for index in "${!MODELS[@]}";do
    name="${NAMES[index]}"
    model="${MODELS[index]}"
    epoch=${EPOCHS[index]}

    for ((idx=1; idx<=${k_fold}; idx++)); do
        echo "--------------------------- Fold ${idx} ---------------------------"

        train_path=${cross_validation_folder}/fold-${idx}/train.tsv
        dev_path=${cross_validation_folder}/fold-${idx}/test.tsv
        res_path=${cross_validation_folder}/fold-${idx}/

        echo Starting: "$name"_"$idx"
        python train.py -model ${model} -train ${train_path} -dev ${dev_path} -res ${res_path} -max_sequence_len 64 -epoch ${epoch} -train_batch_size 16 -valid_batch_size 16 -lr 2e-5 -n_warmup_steps 0 > out_"$idx".txt 2>&1
        
        sleep 2
        mv ${result_path}/PACT.csv ${result_path}/"$name"_"$idx"_PACT.csv
        mv out_"$idx".txt ${result_path}/"$name"_"$idx"_out.txt         
        sleep 2
    done
done

# Deactivate conda env
conda deactivate

echo "Finished: Exiting!"
#Done
