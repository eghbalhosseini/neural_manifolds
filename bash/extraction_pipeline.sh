#!/bin/bash
MODEL_DIR=$1
ANALYZE=$2
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted
PTH_FILE="master_${MODEL_DIR}.csv"
MODEL_ID=$MODEL_DIR
ANALYZE_ID=$ANALYZE
FULL_FILE="${ROOT_DIR}/${MODEL_DIR}/${PTH_FILE}"
echo $FULL_FILE

LINE_COUNT=0
START_INDEX=1
ARRAY_INDEX=0
# separate data to batch of 100 and run them.
while read line; do
	LINE_COUNT=$(expr ${LINE_COUNT} + 1)
	if [ "$(expr ${LINE_COUNT} % 100)" = "0" ]
	then
		echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
		sbatch --array=0-99 --mem 5G -p normal extraction_script.sh ${ARRAY_INDEX} ${MODEL_ID} ${ANALYZE_ID}
		START_INDEX=$(expr ${LINE_COUNT} + 1)
		ARRAY_INDEX=$(expr ${ARRAY_INDEX} + 1)
	fi
done < $FULL_FILE
# run the remaining files
if [ "${LINE_COUNT}" -ge "${START_INDEX}" ]
then
	DIFF=$(expr ${LINE_COUNT} - ${START_INDEX} + 1)
	echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
	# TODO make the DIFF to DIFF -1
	sbatch --array=0-$(expr ${DIFF} - 1) --mem 5G -p normal extraction_script.sh ${ARRAY_INDEX} ${MODEL_ID} ${ANALYZE_ID}
	#bash extraction_script.sh ${ARRAY_INDEX} ${FULL_FILE} ${PKL_FILE} ${ROOT_DIR}
fi
