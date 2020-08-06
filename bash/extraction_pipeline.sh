#!/bin/bash

ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted
PTH_FILE="master_NN-partition_nclass=50_nobj=50000_beta=0.01_sigma=1.5_nfeat=3072-train_test-test_performance.txt"
MODEL_ID='[NN]-[partition/nclass=50/nobj=50000/beta=0.01/sigma=1.5/nfeat=3072]-[train_test]-[test_performance]'
ANALYZE_ID='[mftma]-[exm_per_class=20]-[proj=False]-[rand=True]-[kappa=0]-[n_t=300]-[n_rep=1]'

FULL_FILE="${ROOT_DIR}/${PTH_FILE}"
echo $FULL_FILE

LINE_COUNT=0
START_INDEX=1
ARRAY_INDEX=0
# separate data to batch of 100 and run them.
while read line; do
	LINE_COUNT=$(expr ${LINE_COUNT} + 1)
	if [ "$(expr ${LINE_COUNT} % 50)" = "0" ]
	then
		echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
		sbatch --array=1-50 --mem 8G -p normal extraction_script.sh ${ARRAY_INDEX} ${MODEL_ID} ${ANALYZE_ID}
		#bash extraction_script.sh ${ARRAY_INDEX} ${FULL_FILE} ${PKL_FILE} ${ROOT_DIR}
		START_INDEX=$(expr ${LINE_COUNT} + 1)
		ARRAY_INDEX=$(expr ${ARRAY_INDEX} + 1)
	fi
done < $FULL_FILE
# run the remaining files
if [ "${LINE_COUNT}" -ge "${START_INDEX}" ]
then
	DIFF=$(expr ${LINE_COUNT} - ${START_INDEX} + 1)
	echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
	sbatch --array=1-${DIFF} --mem 8G -p normal extraction_script.sh ${ARRAY_INDEX} ${MODEL_ID} ${ANALYZE_ID}
	#bash extraction_script.sh ${ARRAY_INDEX} ${FULL_FILE} ${PKL_FILE} ${ROOT_DIR}
fi
