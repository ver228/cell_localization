#!/bin/bash

#-P rittscher.prjc -q gpu8.q -l h_vmem=64G -pe ramdisk 2 -l gpu=1 -l gputype=p100
#module use -a /mgmt/modules/eb/modules/all
#module load Anaconda3/5.1.0

echo "********"
source activate pytorch-0.4.1
cd $HOME/GitLab/cell_localization/scripts


python -W ignore train_bbox.py \
--batch_size 64  \
--data_type 'eggs' \
--loss_type 'focal' \
--model_name 'retinanet-resnet34' \
--lr 1e-4 \
--num_workers 4  \
--n_epochs 2000 \
--save_frequency 100 \
--root_data_dir '/tmp/avelino/eggs/'