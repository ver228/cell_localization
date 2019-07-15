#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 64  \
--data_type 'bladder-tiles-no-border-20x' \
--model_name 'unet' \
--loss_type 'l1smooth' \
--lr 64e-6 \
--num_workers 4  \
--is_preloaded True \
--hard_mining_freq 5 \
--n_epochs 2000 \
--save_frequency 200