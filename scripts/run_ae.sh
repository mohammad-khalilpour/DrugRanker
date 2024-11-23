#!/bin/bash

outdim=(128)
batch=(64)
data=$1
expt_dir="expts/ae/LCO/${data}"
fold=0
gexp_file="data/CCLE/CCLE_expression.csv"
ae_ind=19177
if [[ $data == 'ctrpv2' ]]; then
	ae_ind=17743
	gexp_file="data/Combined/combined_rnaseq_data"
fi

# separate pretrained model for each fold in LCO experiments
# for fold in $(seq 0 4); do
# for bs in ${batch[@]}; do
# 	for outd in ${outdim[@]}; do
# 		save_dir="${expt_dir}/all_bs_${bs}_outd_${outd}/fold_${fold}/"
# 		mkdir -p $save_dir
# 		python3 src/train_ae.py --genexp_file $gexp_file --splits_path data/${data}/LCO/pletorg/fold_${fold}/ \
# 		--save_path $save_dir --ae_out_size $outd --bs $bs --cuda > $save_dir/train.log
# 	done
# done
# done


# to pretrain single GeneAE model using all cell lines in LRO experiments
outd=128
bs=64
expt_dir="expts/ae/LRO/${data}"
save_dir="${expt_dir}/all_bs_${bs}_outd_${outd}/"
mkdir -p $save_dir
python3 src/train_ae.py --genexp_file $gexp_file --save_path $save_dir --ae_out_size $outd --bs $bs --use_all --cuda > $save_dir/train.log