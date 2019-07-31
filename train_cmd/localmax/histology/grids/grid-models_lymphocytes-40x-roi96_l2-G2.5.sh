#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts

declare -a MODELS=( "unet-deeper5" "unet-input-halved" "unet-simple"  "unet-attention" "unet-SE" )

for model in "${MODELS[@]}"
do

python -W ignore train_locmax.py \
--batch_size 96  \
--data_type 'lymphocytes-40x' \
--roi_size 96 \
--model_name $model \
--loss_type "l2-G2.5" \
--lr 96e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 60

done