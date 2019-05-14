#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py --batch_size 64 \
--data_type 'bladder-cancer-tils-40x' --roi_size 128 \
--model_name 'unet-bn' --loss_type 'adjl1smooth' \
--lr 64e-5 --num_workers 8  --n_epochs 500 --save_frequency 50