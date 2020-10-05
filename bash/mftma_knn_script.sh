#!/bin/bash
#
#SBATCH -c 8
#SBATCH --exclude node[017-018]
#SBATCH -t 23:00:00

ARRAY_ID=$1
MODEL_ID=$2
ANALYZE_ID=$3
OVERWRITE=$4
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$1        # Taking the task ID as an input parameter.
fi


echo "Line ${JID}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/mftma_analysis.py ${FILE_LINE} ${MODEL_ID} ${ANALYZE_ID} ${OVERWRITE}

