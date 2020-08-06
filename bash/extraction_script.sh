#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 5000
ARRAY_ID=$1
MODEL_ID=$2
ANALYZE_ID=$3
#
let FILE_LINE=(50*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)
echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
python ~/MyCodes/neural_manifolds/extract_data.py ${FILE_LINE} ${MODEL_ID} ${ANALYZE_ID}