#!/bin/bash

genexp_path='data/CCLE/CCLE_expression.csv'
log_steps=5
max_iter=100
num_folds=1
device=${1:-'cuda:0'}

models=("lambdaloss")
representations=("morgan_count")
setups=("LCO")

data_set="ctrp"
data_dir="data/$data_set"

for setup in "${setups[@]}"; do
    for model in "${models[@]}"; do
        for representation in "${representations[@]}"; do
            for fold in $(seq 0 $((num_folds-1))); do
                if [[ $setup == 'LCO' ]]; then
                    ae_path="expts/ae/$setup/$data_set/all_bs_64_outd_128/fold_$fold/model.pt"
                    splits_path="$data_dir/$setup/pletorg/"
                elif [[ $setup == 'LRO' ]]; then
                    ae_path="expts/ae/$setup/$data_set/all_bs_64_outd_128/model.pt"
                    splits_path="$data_dir/$setup/"
                fi
                save_dir="expts/result_gnn/$setup/$data_set/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/get_results.py \
                    --model "$model" \
                    --only_fold $fold \
                    --data_path "$data_dir/$setup/aucs.txt" \
                    --smiles_path "$data_dir/cmpd_smiles.txt" \
                    --splits_path $splits_path \
                    --save_path $save_dir \
                    --pretrained_ae \
                    --trained_ae_path "$ae_path" \
                    --feature_gen "$representation" \
                    --gnn "dmpn" \
                    --max_iter $max_iter \
                    --desired_device $device \
                    --genexp_path "$genexp_path" \
                    --setup "$setup" \
                    --checkpointing \
                    --log_steps $log_steps \
                    --get_results_gnn "model-inspection"
            done
        done
    done
done