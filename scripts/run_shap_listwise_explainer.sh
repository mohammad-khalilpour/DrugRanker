#!/bin/bash

num_folds=5

models=("lambdaloss")
representations=("rdkit2d_atompair")
described_features=("cell_line")

data_set="ctrp"
data_dir="data/$data_set"


genexp_path='data/CCLE/CCLE_expression.csv'
selected_genes_indices="data/selected_genes_indices_$data_set.npy"

cancer_type="Breast Cancer"
background_samples=2

cancers_types_file="data/$data_set/LRO/cancer.txt"
cell_lines_file="data/$data_set/LRO/cells.txt"

for model in "${models[@]}"; do
    for representation in "${representations[@]}"; do
        for described_feature in "${described_features[@]}"; do
            for fold in $(seq 0 $((num_folds-1))); do
                save_dir="saved_models/shapley/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p "$save_dir"
                mkdir -p "$log_dir"
                python3 src/shapley_listwise.py \
                    --model "$data_set/$model/$representation/$fold" \
                    --described_features "$described_feature" \
                    --data_path "$data_dir/LRO/aucs.txt" \
                    --smiles_path "$data_dir/cmpd_smiles.txt" \
                    --save_path "$save_dir" \
                    --feature_gen "$representation" \
                    --genexp_path "$genexp_path" \
                    --selected_genes_indices "$selected_genes_indices" \
                    --cancer_type "$cancer_type" \
                    --cancers_types_file "$cancers_types_file" \
                    --cell_lines_file "$cell_lines_file" \
                    --background_samples "$background_samples" \
                    --device "cpu"
            done
        done
    done
done