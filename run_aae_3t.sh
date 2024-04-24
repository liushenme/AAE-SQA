#!/bin/bash

# Exit on error
set -e
set -o pipefail

stage=0
tag=""  # Controls the directory name associated to the experiment
id=$CUDA_VISIBLE_DEVICES

# Experiment config
is_complex=false

datadir="bit"

# Evaluation
eval_use_gpu=1
model="aae_3t_deep"
. ./utils/parse_options.sh

tag=${datadir}_${tag}
expdir=exp/train_${tag}
mkdir -p $expdir && echo $uuid >> $expdir/run_uuid.txt
echo "Results from the following experiment will be stored in $expdir"


mkdir -p logs
CUDA_VISIBLE_DEVICES=$id python -u train_e2e_aae_3t.py \
	--is_complex $is_complex \
	--model_type $model \
	--exp_dir ${expdir}/ | tee logs/train_${tag}.log
cp logs/train_${tag}.log $expdir/train.log


