#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1

export PATH="/well/rittscher/projects/base/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 32  \
--data_type 'eggsadam-stacked' \
--roi_size 48 \
--model_name 'unetv2b-tanh' \
--loss_type 'maxlikelihood' \
--lr 32e-6 \
--num_workers 2  \
--is_preloaded True \
--hard_mining_freq 10 \
--n_epochs 1000