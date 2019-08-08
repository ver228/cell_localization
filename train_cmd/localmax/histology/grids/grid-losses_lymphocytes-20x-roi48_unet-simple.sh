#!/bin/bash
#$ -P rittscher.prjc -q gpu8.q -pe shmem 1 -l gpu=1


export PATH="/users/rittscher/avelino/miniconda3/bin:$PATH"

echo "********"
source activate pytorch-1.0
cd $HOME/GitLab/cell_localization/scripts
#declare -a LOSSES=( "l2-reg-G1.5" "l2-reg-G2.5" "l1smooth-reg-G1.5" "l1-reg-G1.5" "l2-G2.5" "maxlikelihood" "l2-G1.5" )
declare -a LOSSES=( "l2-G2.5" "maxlikelihood" "l2-G1.5" "l1smooth-G2.5" "l1-G2.5" )

for loss in "${LOSSES[@]}"
do

python -W ignore train_locmax.py \
--batch_size 256  \
--data_type 'lymphocytes-20x' \
--roi_size 48 \
--model_name 'unet-simple' \
--loss_type $loss \
--lr 256e-6 \
--num_workers 1  \
--is_preloaded True \
--n_epochs 60

done