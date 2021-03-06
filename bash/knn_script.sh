#!/bin/bash
#
#SBATCH -c 8
#SBATCH --exclude node[017-018]
#SBATCH -t 3:00:00


GRAND_FILE=$1
#MODEL_ID=$2
#ANALYZE_ID=$3
OVERWRITE='false'
#
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2        # Taking the task ID as an input parameter.
fi
echo "${GRAND_FILE}"

while IFS=, read -r line_count model_line model analyze layer k dist_metric num_subsamples ; do
  #echo "line_count ${model}"
  if [ $JID == $line_count ]
    then
      echo "found the right match ${line_count}"
      run_model_line=$model_line
      run_model=$model
      run_analyze=$analyze
      run_layer=$layer
      run_k=$k
      run_dist_metric=$dist_metric
      run_num_subsamples=$num_subsamples
      do_run=true
      break
    else
      do_run=false
      #echo "didnt the right match"
  fi

done <"${GRAND_FILE}"

echo "model ${model}"
echo "analyze ${run_analyze}"
echo "layer to analyze ${run_layer}"



module add mit/matlab/2020a
matlab -nodisplay -r "addpath(genpath('/om/user/${USER}/neural_manifolds/'));\
runKNN('root_dir','$ROOT_DIR','analyze_identifier','$run_analyze','model_identifier','$run_model','layer','$run_layer','dist_metric','$run_dist_metric','k',$run_k,'num_subsamples',$run_num_subsamples);\
fprintf('done');quit;"