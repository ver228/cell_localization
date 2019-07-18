#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 128  \
--data_type 'eggsadam' \
--roi_size 48 \
--model_name 'unetv2b' \
--loss_type 'l1smooth' \
--lr 128e-6 \
--num_workers 4  \
--is_preloaded True \ \
--hard_mining_freq 1 \
--n_epochs 1000