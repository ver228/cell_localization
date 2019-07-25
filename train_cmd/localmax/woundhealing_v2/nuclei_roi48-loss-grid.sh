#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts

declare -a LOSSES=("l2-G1.5" "l2-reg-G1.5" "l2-reg-G2.5" "l1smooth-reg-G1.5" "l1-reg-G1.5") #"maxlikelihood" 

for loss in "${LOSSES[@]}"
do

python -W ignore train_locmax.py \
--batch_size 256 \
--data_type 'woundhealing-v2-nuclei' \
--loss_type $loss \
--model_name 'unet-simple' \
--roi_size 48 \
--lr 256e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 50 

done
