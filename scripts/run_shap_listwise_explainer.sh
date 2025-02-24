#!/bin/bash

genexp_path="../data/CCLE/CCLE_expression.csv"

num_folds=5

# models=("pairpushc" "lambdaloss" "lambdarank" "neuralndcg" "listone" "listall")
# representations=('morgan_count' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')
# setups=("LCO" "LRO")

models=("lambdaloss")
representations=("rdkit2d_atompair")
described_features=("cell_line")

data_set="ctrp"
data_dir="data/$data_set"


genexp_path='data/CCLE/CCLE_expression.csv'
selected_genexp_path="data/${data_set}/selected_genes_indices_${data_set}.npy"

for model in "${models[@]}"; do
    for representation in "${representations[@]}"; do
        for described_feature in "${described_features[@]}"; do
            for fold in $(seq 0 $((num_folds-1))); do
                save_dir="expts/result_shap_expl/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/shapley_listwise.py \
                    --model "$model" \
                    --test_numbers 100 \
                    --described_features "$described_feature" \
                    --fold_num $fold \
                    --data_path "$data_dir/LRO/aucs.txt" \
                    --smiles_path "$data_dir/cmpd_smiles.txt" \
                    --save_path "$save_dir" \
                    --feature_gen "$representation" \
                    --genexp_path "$genexp_path" \
                    --selected_genexp_path "$selected_genexp_path" \
                    --device "cpu"
            done
        done
    done
done