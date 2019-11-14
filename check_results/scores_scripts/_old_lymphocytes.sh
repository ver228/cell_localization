#!/bin/bash

#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1
export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0

python $HOME/GitLab/cell_localization/check_results/finetune_validation.py \
--data_type 'lymphocytes-20x' \
--flow_type 'lymphocytes' \
--root_model_dir  $HOME'/workspace/localization/results/locmax_detection/lymphocytes/20x/' \
--checkpoint_name 'model_best.pth.tar'