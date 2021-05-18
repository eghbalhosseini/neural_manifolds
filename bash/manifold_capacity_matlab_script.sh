#!/bin/bash
#
#SBATCH -c 8
#SBATCH --exclude node[017-018]
#SBATCH -t 5:00:00

GRAND_FILE=$1
OVERWRITE='false' # or 'true'
#

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
  JID=$SLURM_ARRAY_TASK_ID    # Taking the task ID in a job array as an input parameter.
else
  JID=$2       # Taking the task ID as an input parameter.
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

echo "line ${run_model_line}"
echo "model ${run_model}"
echo "analyze ${run_analyze}"
echo "file to analyze ${run_file}"
module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

# implement the code in matlab
module add mit/matlab/2018b
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);\
addpath(genpath('/om/user/${USER}/neural_manifolds/matlab/'));\
save_path='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/';\
load(erase('${run_file}',' '));\
X = (activation.projection_results{1}.(activation.layer_name));\
for ii=1:min(30,size(X,1)),XtotT{ii} = double(squeeze(X(ii,:,1:20))); end ;\
options.n_rep =10;\
options.seed0 = 1;\
options.flag_NbyM =1;\
[output] = manifold_simcap_analysis(XtotT, options);\
save_id=strrep(erase('${run_file}',' '),'extracted_v3.mat','capacity_v3.mat');\
save(save_id,'output');\
quit;"

