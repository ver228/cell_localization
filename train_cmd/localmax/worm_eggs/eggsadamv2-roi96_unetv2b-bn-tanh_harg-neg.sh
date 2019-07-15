#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1

module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 64  \
--data_type 'eggsadamv2' \
--roi_size 96 \
--model_name 'unetv2b-bn-tanh' \
--loss_type 'l1smooth' \
--lr 64e-6 \
--num_workers 1  \
--is_preloaded True \
--hard_mining_freq 20 \
--n_epochs 1000