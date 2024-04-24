Code for the TNNLS paper 'Non-intrusive Speech Quality Assessment Based on Deep Neural Networks for Speech Communication'

## Pre-training:
./run_aae_3t.sh --datadir xxx --stage 0 --tag daae_3t --id 0

## Fine-tuning:
./run_ft_aae_3t.sh --datadir qq --predir xxx --stage 0 --tag daae_3t_ft --id 0

## Quality assessement:
./run_fu.sh --datadir qq --featdir xxx --stage 0 --tag xxx --id 0

