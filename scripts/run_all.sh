#!/bin/bash


DATA_FOLDER="data/ctrp"
DATA_SET="ctrp"
genexp_path='data/CCLE/CCLE_expression.csv'
save_path="expts/result/LCO/$DATA_SET/"

# models=("pairpushc" "lambdaloss" "lambdarank" "neuralndcg" "listone" "listall")
# representations=('morgan_count' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')
# setups=("LCO" "LRO")

models=("neuralndcg")
representations=("layered_rdkit")
setups=("LCO")

for setup in "${setups[@]}"; do
    for model in "${models[@]}"; do
        for representation in "${representations[@]}"; do
            for fold in $(seq 0 0); do
                ae_path="expts/ae/LCO/$DATA_SET/all_bs_64_outd_128/fold_$fold/model.pt"
                save_dir="expts/result/$setups/$DATA_SET/$model/$representation/"
                log_dir=$save_dir/logs/
                mkdir -p $log_dir
                python3 src/cross_validate.py \
                    --model "$model" \
                    --only_fold $fold \
                    --data_path "$DATA_FOLDER/LCO/aucs.txt" \
                    --smiles_path "$DATA_FOLDER/cmpd_smiles.txt" \
                    --splits_path "$DATA_FOLDER/LCO/pletorg/" \
                    --save_path $save_dir \
                    --pretrained_ae \
                    --trained_ae_path "$ae_path" \
                    --feature_gen "$representation" \
                    --max_iter "1" \
                    --genexp_path "$genexp_path" \
                    --setup "$setup" \
                    --log_steps 1 > $log_dir/avg_results.txt
            done
        done
    done
done