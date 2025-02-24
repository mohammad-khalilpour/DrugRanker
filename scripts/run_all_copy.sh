#!/bin/bash

log_steps=5
max_iter=500
num_folds=5
device=${1:-'cuda:0'}

# models=("pairpushc" "lambdaloss" "lambdarank" "neuralndcg" "listone" "listall")
# representations=('morgan_count' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')
# setups=("LCO" "LRO")

models=("lambdarank")
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
                save_dir="expts/result_expl/drug_atten_norm/$setup/$data_set/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/cross_validate.py \
                    --model "$model" \
                    --only_fold $fold \
                    --data_path "$data_dir/$setup/aucs.txt" \
                    --smiles_path "$data_dir/cmpd_smiles.txt" \
                    --splits_path $splits_path \
                    --save_path $save_dir \
                    --to_use_ae_emb \
                    --pretrained_ae \
                    --trained_ae_path "$ae_path" \
                    --feature_gen "$representation" \
                    --max_iter $max_iter \
                    --update_emb "drug-attention" \
                    --desired_device $device \
                    --genexp_path "$genexp_path" \
                    --selected_genexp_path "$selected_genexp_path" \
                    --checkpointing \
                    --to_save_attention_weights \
                    --to_save_best \
                    --setup "$setup" \
                    --log_steps $log_steps > $log_dir/results_$((fold+1)).txt
            done
            python3 src/get_results.py \
                --save_path $save_dir \
                --log_steps $log_steps
        done
    done
done