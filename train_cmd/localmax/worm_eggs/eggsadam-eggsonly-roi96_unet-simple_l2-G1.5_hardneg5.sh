#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 128 \
--data_type 'worm-eggs-adam' \
--loss_type 'l2-G1.5' \
--model_name 'unet-simple' \
--flow_type 'eggsonly' \
--roi_size 96 \
--lr 128e-6 \
--num_workers 1  \
--is_preloaded True \
--hard_mining_freq 5 \
--n_epochs 150