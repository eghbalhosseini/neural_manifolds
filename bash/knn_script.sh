#!/bin/bash
#
#SBATCH -c 8
#SBATCH --exclude node[017-018]
#SBATCH -t 3:00:00


GRAND_FILE=$1
#MODEL_ID=$2
#ANALYZE_ID=$3
OVERWRITE='true'
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2        # Taking the task ID as an input parameter.
fi
echo "${GRAND_FILE}"

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

echo "model ${run_model}"
echo "analyze ${run_analyze}"
echo "file to analyze ${run_file}"



module add mit/matlab/2020a
matlab -nodisplay -r "addpath(genpath('/om/user/`whoami`/neural_manifolds/'));knn_analysis(char(${run_file}),char(${run_model}),char(${run_analyze});quit;"

