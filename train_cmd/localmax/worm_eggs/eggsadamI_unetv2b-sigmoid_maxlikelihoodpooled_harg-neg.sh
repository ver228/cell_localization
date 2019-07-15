#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1

export PATH="/well/rittscher/projects/base/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 256  \
--data_type 'eggsadamI' \
--roi_size 48 \
--model_name 'unetv2b-sigmoid' \
--loss_type 'maxlikelihoodpooled' \
--lr 128e-6 \
--num_workers 1  \
--is_preloaded True \
--hard_mining_freq 10 \
--n_epochs 1000