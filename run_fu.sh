#!/bin/bash

# Exit on error
set -e
set -o pipefail

stage=0
tag=""  # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES

# Experiment config
is_complex=false  # If we use a complex network for training.

datadir="bit"

# Evaluation
eval_use_gpu=1
model_sq="cnn_lstm_4d_fu"
featdir=exp/train_qq_aae_3t_ft

. ./utils/parse_options.sh


tag=${datadir}_${tag}
expdir=exp/train_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Train"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path -u train_e2e_fu.py \
		--is_complex $is_complex \
		--model_type $model_sq \
		--feat_dir $featdir \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
		#--data $datadir \
		#--json_dir $dumpdir \
	cp logs/train_${tag}.log $expdir/train.log
fi


#exit 0;

if [[ $stage -le  1 ]]; then
  echo "Stage 1 : Evaluate"
  CUDA_VISIBLE_DEVICES=$id $python_path -u pytorch/evaluate_e2e_fu.py \
		--json_dir dataset_own/qq \
		--use_gpu $eval_use_gpu \
		--feat_dir $featdir \
		--exp_dir $expdir/ | tee logs/eval_${tag}.log
		#--json_dir dataset_own/WB_ACR \
		#--data $datadir \
		#--json_dir dataset_own/NB_ACR \
	cp logs/eval_${tag}.log $expdir/eval.log
fi

