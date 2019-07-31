#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0

python $HOME/GitLab/cell_localization/check_results/finetune_validation.py \
--data_type 'eosinophils-20x' \
--flow_type 'eosinophils' \
--root_model_dir  $HOME'/workspace/localization/results/locmax_detection/eosinophils/20x/eosinophils-20x/different_models' \
--checkpoint_name 'model_best.pth.tar'