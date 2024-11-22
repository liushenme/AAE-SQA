#!/bin/bash

# Exit on error
set -e
set -o pipefail

stage=0
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

# Experiment config
is_complex=false  # If we use a complex network for training.

datadir="bit"

# Evaluation
eval_use_gpu=1
. ./utils/parse_options.sh

model_sq="MOSNet"

if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Train Speech Quality"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path -u train_e2e.py \
		--is_complex $is_complex \
		--model_type $model_sq \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
		#--data $datadir \
		#--json_dir $dumpdir \
	cp logs/train_${tag}.log $expdir/train.log
fi

if [[ $stage -le  1 ]]; then
  echo "Stage 1 : Evaluate Speech Quality"
  CUDA_VISIBLE_DEVICES=$id $python_path -u pytorch/evaluate_e2e.py \
		--json_dir dataset_own/WB_ACR \
		--use_gpu $eval_use_gpu \
		--exp_dir $expdir/ | tee logs/eval_${tag}.log
		#--json_dir dataset_own/qq \
		#--data $datadir \
		#--json_dir dataset_own/NB_ACR \
	cp logs/eval_${tag}.log $expdir/eval.log
fi

