#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 5000

ARRAY_ID=$1
PARAMETER_FILE=$2
PICKLE_FILE=$3
ROOT_DIR=$4

let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})


python /Users/eghbalhosseini/MyCodes/neural_manifolds/extract_data.py ${ROOT_DIR} ${PICKLE_FILE} ${PARAMETERS} 20
#bash openmind_sample.sh ${PARAMETERS}