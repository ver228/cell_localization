#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 64 \
--data_type 'woundhealing-v2-mix' \
--loss_type 'l1smooth' \
--model_name 'unetv2b-init-normal' \
--roi_size 96 \
--lr 64e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 1000 \
--save_frequency 100