#!/bin/bash

DATA_FOLDER="../data/ctrp"
ae_path="../expts/ae/LCO/ctrp/all_bs_64_outd_128/fold_2/model.pt"

models=("listone" "listall" "pairpushc" "lambdaloss" "lambdarank" "neuralndcg")

representations=('morgan_count' 'map4' 'avalon' 'atom_pair' '2d_pharmacophore' 'layered_rdkit')

for model in "${models[@]}"; do
    for representation in "${representations[@]}"; do
        python src/cross_validate.py \
            --model "$model" \
            --data_path "$DATA_FOLDER/LCO/aucs.txt" \
            --smiles_path "$DATA_FOLDER/cmpd_smiles.txt" \
            --splits_path "$DATA_FOLDER/LCO/pletorg/" \
            --pretrained_ae \
            --ae_path "$ae_path" \
            --fgen "$representation" \
            --setup "LCO"
    done
done
