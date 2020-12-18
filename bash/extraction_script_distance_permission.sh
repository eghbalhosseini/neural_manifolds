#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -t 1:00:00
#SBATCH --mem=8000

GRAND_FILE=$1
OVERWRITE='false'

#
# this is where we put those lines , what lines ?!!

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2        # Taking the task ID as an input parameter.
fi
echo "${GRAND_FILE}"
echo $JID

while IFS=, read -r line_count model_line model analyze analyze_file ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"
      run_model_line=$model_line
      run_model=$model
      run_analyze=$analyze
      run_file=$analyze_file
      do_run=true
      break
    else
      do_run=false
      #echo "didnt the right match"
  fi

done <"${GRAND_FILE}"

#echo "line ${run_model_line}"
#echo "model ${run_model}"
#echo "analyze ${run_analyze}"
#echo "file to analyze ${run_file}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec -B /om:/om,/mindhive:/mindhive /om/user/${USER}/simg_images/neural_manifolds_tiny.simg ./om/user/${USER}/neural_manifolds/bash/extract_distance_permission.sh ${run_file} ${run_model_line} ${run_model} ${run_analyze} ${OVERWRITE}