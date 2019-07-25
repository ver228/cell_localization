#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts

declare -a MODELS=("unet-simple-bn" "unet-attention" "unet-SE" "unet-flat" "unet-wide" "unet-simple")

for model in "${MODELS[@]}"
do

python -W ignore train_locmax.py \
--batch_size 256  \
--data_type 'lymphocytes-20x' \
--roi_size 48 \
--model_name $model \
--loss_type "maxlikelihood" \
--lr 256e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 60

done