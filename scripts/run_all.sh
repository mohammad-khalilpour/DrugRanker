#!/bin/bash


DATA_FOLDER="data/prism"
DATA_SET="prism"
genexp_path='data/CCLE/CCLE_expression.csv'
save_path="expts/result/LCO/$DATA_SET/"
log_steps=5
max_iter=100
num_folds=5
device=${1:-'cuda:0'}

# models=("pairpushc" "lambdaloss" "lambdarank" "neuralndcg" "listone" "listall")
# representations=('morgan_count' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')
# setups=("LCO" "LRO")

models=("lambdaloss" "lambdarank")
representations=("morgan_count")
setups=("LRO")

for setup in "${setups[@]}"; do
    for model in "${models[@]}"; do
        for representation in "${representations[@]}"; do
            for fold in $(seq 0 $((num_folds-1))); do
                if [[ $setup == 'LCO' ]]; then
                    ae_path="expts/ae/$setup/$DATA_SET/all_bs_64_outd_128/fold_$fold/model.pt"
                    splits_path="$DATA_FOLDER/$setup/pletorg/"
                elif [[ $setup == 'LRO' ]]; then
                    ae_path="expts/ae/$setup/$DATA_SET/all_bs_64_outd_128/model.pt"
                    splits_path="$DATA_FOLDER/$setup/"
                fi
                save_dir="expts/result/$setup/$DATA_SET/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/cross_validate.py \
                    --model "$model" \
                    --only_fold $fold \
                    --data_path "$DATA_FOLDER/$setup/aucs.txt" \
                    --smiles_path "$DATA_FOLDER/cmpd_smiles.txt" \
                    --splits_path $splits_path \
                    --save_path $save_dir \
                    --pretrained_ae \
                    --trained_ae_path "$ae_path" \
                    --feature_gen "$representation" \
                    --max_iter $max_iter \
                    --desired_device $device \
                    --genexp_path "$genexp_path" \
                    --setup "$setup" \
                    --log_steps $log_steps > $log_dir/results_$((fold+1)).txt
            done
            python3 src/get_results.py \
                --save_path $save_dir \
                --log_steps $log_steps
        done
    done
done