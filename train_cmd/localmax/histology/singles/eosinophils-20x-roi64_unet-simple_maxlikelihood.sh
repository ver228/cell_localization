#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 256  \
--data_type 'eosinophils-20x' \
--roi_size 64 \
--model_name 'unet-simple' \
--loss_type 'maxlikelihood' \
--lr 256e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 100