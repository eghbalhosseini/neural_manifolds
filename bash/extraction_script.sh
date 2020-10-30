#!/bin/bash
#
#SBATCH -N 1 # on one node
#SBATCH -t 2:00:00
ARRAY_ID=$1
MODEL_ID=$2
ANALYZE_ID=$3
#
# this is where we put those lines ,
let FILE_LINE=(100*$ARRAY_ID + $SLURM_ARRAY_TASK_ID)
echo "Line ${FILE_LINE}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec -B /om:/om /mindhive:/mindhive /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/extract_data.py ${FILE_LINE} ${MODEL_ID} ${ANALYZE_ID}