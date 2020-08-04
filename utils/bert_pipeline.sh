#!/bin/bash
PARAMETER_FILE="pos_parameters2.txt"

#Python Image

ROOT_DIR=/om2/user/hangle/CWR_manifold/manifold_recipe/task_files
SELE_TYPE="curated" # Defines the selection algorithm by which we extracted the labels.
DSET_TYPE="ptb-punct" # Defines the type of dataset we will use. ptb-punct is the dataset with punctuation while ptb-no-punct is without.
MANIFOLDS=("pos" "word" "sem-tag" "dep-depth") # Represents the different tasks we are working on
MASKED=("Mask-False" "Mask-True") # Represents Masked (True) or Normal (False) sentences
ANALYSIS=("features") #Type of result analysis to load
MODELS=("bert-base-cased") #Represent The model we want to analyze
LAYERS=$(seq 0 12) #Layer indices for BERT
SEEDS=$(seq 0 4)

> ${PARAMETER_FILE}
for TYPE in ${SELE_TYPE[@]};do
	for DSET in ${DSET_TYPE[@]};do
		for MANI in ${MANIFOLDS[@]};do
			for MODEL in ${MODELS[@]};do
				for MASK in ${MASKED[@]};do
					for ANA in ${ANALYSIS[@]};do
						for LAYER in ${LAYERS[@]};do
							for SEED in ${SEEDS[@]};do
								echo "${ROOT_DIR} ${TYPE} ${DSET} ${MANI} ${MODEL} ${MASK} ${ANA} ${LAYER} ${SEED}" >> ${PARAMETER_FILE}
							done
						done
					done
				done
			done
		done
	done
done

# 2. Push all the Jobs through Job Arrays

# 2.a Iterate through parameter lines. 
# For every 100 lines we should create a new job array

LINE_COUNT=0
START_INDEX=1
ARRAY_INDEX=0
while read line; do
	LINE_COUNT=$(expr ${LINE_COUNT} + 1)

	if [ "$(expr ${LINE_COUNT} % 100)" = "0" ]
	then
		echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
		sbatch --array=1-100 --mem 8G -p normal openmind_script.sbatch ${ARRAY_INDEX} ${PARAMETER_FILE}
		START_INDEX=$(expr ${LINE_COUNT} + 1)
		ARRAY_INDEX=$(expr ${ARRAY_INDEX} + 1)
	fi

done < ${PARAMETER_FILE}

# 2.b Check to make sure that all parameters were tested
if [ "${LINE_COUNT}" -ge "${START_INDEX}" ]
then
	DIFF=$(expr ${LINE_COUNT} - ${START_INDEX} + 1)
	echo "New Array For Parameters from ${START_INDEX} to ${LINE_COUNT}"
	sbatch --array=1-${DIFF} --mem 8G -p normal openmind_script.sbatch ${ARRAY_INDEX} ${PARAMETER_FILE}
fi
