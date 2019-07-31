#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0

python $HOME/GitLab/cell_localization/check_results/finetune_validation.py \
--data_type 'worm-eggs-adam' \
--flow_type 'eggs' \
--root_model_dir  $HOME'/workspace/localization/results/locmax_detection/eggs/worm-eggs-adam/' \
--checkpoint_name 'model_best.pth.tar'