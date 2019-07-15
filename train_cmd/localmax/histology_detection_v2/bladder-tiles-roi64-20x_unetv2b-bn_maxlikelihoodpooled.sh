#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 256  \
--data_type 'bladder-tiles-20x' \
--roi_size 64 \
--model_name 'unetv2b-bn' \
--loss_type 'maxlikelihoodpooled' \
--lr 256e-5 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 2000 \
--save_frequency 200