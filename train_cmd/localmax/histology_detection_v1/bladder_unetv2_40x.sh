#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 128  \
--data_type 'bladder-40x' \
--model_name 'unetv2' \
--loss_type 'l1smooth' \
--lr 128e-6 \
--num_workers 3  \
--is_preloaded True \
--n_epochs 2000 \
--save_frequency 200