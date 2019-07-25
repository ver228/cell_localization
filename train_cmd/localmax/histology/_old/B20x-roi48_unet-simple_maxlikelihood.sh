#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 512  \
--data_type 'bladder-20x' \
--roi_size 48 \
--model_name 'unet-simple' \
--loss_type 'maxlikelihood' \
--lr 512e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 50 