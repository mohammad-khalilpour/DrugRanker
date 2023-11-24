#!/bin/bash

# genexp_path='data/CCLE/CCLE_expression.csv'
# genexp_path='data/CCLE/CCLE_expression_ctrp_w20.csv'
log_steps=5
max_iter=100
num_folds=5
device=${1:-'cuda:0'}

# models=("pairpushc" "lambdaloss" "lambdarank" "neuralndcg" "listone" "listall")
# representations=('morgan_count' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')
# setups=("LCO" "LRO")

models=("lambdaloss")
representations=("atom_pair")
setups=("LCO")

data_set="ctrp"
data_dir="data/$data_set"
genexp_path='data/CCLE/CCLE_expression.csv'
# genexp_path="data/CCLE/CCLE_expression_${data_set}_w20.csv"
selected_genexp_path="data/${data_set}/selected_genes_indices_${data_set}.npy"

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
                save_dir="expts/result_expl/$setup/$data_set/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/inference.py \
                    --model "$model" \
                    --only_fold $fold \
                    --do_test \
                    --data_path "$data_dir/$setup/aucs.txt" \
                    --smiles_path "$data_dir/cmpd_smiles.txt" \
                    --splits_path $splits_path \
                    --save_path $save_dir \
                    --pretrained_ae \
                    --trained_ae_path "$ae_path" \
                    --feature_gen "$representation" \
                    --max_iter $max_iter \
                    --update_emb "ppi-attention" \
                    --desired_device $device \
                    --genexp_path "$genexp_path" \
                    --selected_genexp_path "$selected_genexp_path" \
                    --setup "$setup" \
                    --log_steps $log_steps 
            done
        done
    done
done