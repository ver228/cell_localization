#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 128  \
--data_type 'heba-int' \
--loss_type 'l1smooth' \
--model_name 'unet' \
--lr 2e-3 \
--num_workers 8  \
--n_epochs 1000 \
--save_frequency 100 \
--hard_mining_freq 5 \
--root_data_dir /tmp/avelino/heba/data