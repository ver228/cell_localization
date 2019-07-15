#### -P rittscher.prjc -q long.qc -pe shmem 2

#$ -P rittscher.prjc
#$ -q short.qc
#$ -pe shmem 1

## works in compG009 and home

source /etc/profile.d/modules.sh
module use -a /mgmt/modules/eb/modules/all
module load Anaconda3/5.1.0

module load LibTIFF/4.0.9-GCCcore-7.3.0

VIPS_DIR=/apps/well/libvips/8.6.3-openslide/bin
APP=vips

DATA_DIR=/well/rittscher/users/avelino/localization/TILS_candidates_4096
RESULT_DIR=/well/rittscher/users/avelino/localization/TILS_candidates_4096/deepzoom

mkdir -p $RESULT_DIR

for FILE in $DATA_DIR/*.png; do
	FILENAME=${FILE##*/}
	BASENAME=${FILENAME%.*}

	$VIPS_DIR/$APP dzsave "$FILE" "$RESULT_DIR/$BASENAME"
    
done
