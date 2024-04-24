#!/bin/bash

# Exit on error
set -e
set -o pipefail

stage=0
tag=""  # Controls the directory name associated to the experiment
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=$CUDA_VISIBLE_DEVICES

is_complex=false  # If we use a complex network for training.

datadir="bit"

# Evaluation
eval_use_gpu=1
model="aae_3t_deep"
predir=exp/train_mi_smi_polqa_ae_aae_3t

. ./utils/parse_options.sh


tag=${datadir}_${tag}
expdir=exp/train_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


if [[ $stage -le  0 ]]; then
  echo "Stage 0 : Train"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path -u train_e2e_ft_aae_3t.py \
		--is_complex $is_complex \
		--model_type $model \
		--pre_dir $predir \
		--exp_dir ${expdir}/ | tee logs/train_${tag}.log
	cp logs/train_${tag}.log $expdir/train.log
fi


if [[ $stage -le  1 ]]; then
  echo "Stage 1 : Evaluate"
  CUDA_VISIBLE_DEVICES=$id $python_path -u pytorch/evaluate_e2e_aae_3t.py \
		--json_dir dataset_own/qq \
		--use_gpu $eval_use_gpu \
		--exp_dir $expdir/ | tee logs/eval_${tag}.log
	cp logs/eval_${tag}.log $expdir/eval.log
fi
