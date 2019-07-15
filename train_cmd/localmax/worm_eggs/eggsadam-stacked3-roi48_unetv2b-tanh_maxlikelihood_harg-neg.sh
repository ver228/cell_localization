#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1

export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 60  \
--data_type 'eggsadam-stacked3' \
--roi_size 48 \
--model_name 'unetv2b-tanh' \
--loss_type 'maxlikelihood' \
--lr 60e-6 \
--num_workers 2  \
--is_preloaded True \
--hard_mining_freq 10 \
--n_epochs 1000