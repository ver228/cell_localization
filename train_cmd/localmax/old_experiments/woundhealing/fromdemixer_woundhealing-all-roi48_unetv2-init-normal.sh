#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_locmax.py \
--batch_size 128  \
--data_type 'woundhealing-all-roi48' \
--loss_type 'l1smooth' \
--model_name 'unetv2-init-normal' \
--model_path_init $HOME'/workspace/localization/results/locmax_detection/woundhealing/woundhealing-demixed-roi48/woundhealing-demixed-roi48_unetv2-init-normal_l1smooth_20190602_112946_adam_lr0.000128_wd0.0_batch128/model_best.pth.tar' \
--lr 128e-6 \
--num_workers 3  \
--is_preloaded True \
--n_epochs 1000 \
--save_frequency 100