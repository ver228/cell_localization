#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 256 \
--data_type 'woundhealing-v2-mix' \
--loss_type 'l2-G1.5' \
--model_name 'unet-simple' \
--optimizer_name 'sgd' \
--lr_scheduler_name 'stepLR-4-0.1' \
--roi_size 48 \
--lr 256e-7 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 16 \
--save_frequency 100