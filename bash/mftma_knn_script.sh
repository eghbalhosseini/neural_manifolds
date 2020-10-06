#!/bin/bash
#
#SBATCH -c 1
#SBATCH --exclude node[017-018]
#SBATCH -t 1:00:00

GRAND_FILE=$1
#MODEL_ID=$2
#ANALYZE_ID=$3
#OVERWRITE=$4
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$1        # Taking the task ID as an input parameter.
fi
echo "Line ${GRAND_FILE}"

while IFS=, read -r line_count model analyze analyze_file ; do
  echo "line_count ${model}"
#  if [ $JID == $line_count ]
#    then
#      echo "found the right match"
#      run_model=$model
#      run_analyze=$analyze
#      run_file=$analyze_file
#      do_run=true
#    else
#      do_run=false
#      echo "didnt the right match"
#  fi

done <$GRAND_FILE

#if [ "$do_run" = true ] ; then
#  echo 'correct parsing'
#fi
#module add openmind/singularity
#export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
#XDG_CACHE_HOME=/om/user/`whoami`/st
#export XDG_CACHE_HOME

# Get the relevant line from the parameters
#singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/mftma_analysis.py ${FILE_LINE} ${MODEL_ID} ${ANALYZE_ID} ${OVERWRITE}

