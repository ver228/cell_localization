#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_demixer.py \
--batch_size 64  \
--data_type 'cell-demixer' \
--loss_type 'l1smooth' \
--roi_size 128 \
--lr 32e-5 \
--num_workers 8  \
--n_epochs 1000 \
--save_frequency 100 \
--is_preloaded True