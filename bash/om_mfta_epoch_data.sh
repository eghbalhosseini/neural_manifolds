#!/bin/sh
#SBATCH --job-name=mftma_epoch
#SBATCH --array=0-39
#SBATCH --time=56:00:00
#SBATCH --ntasks=1
#SBATCH --mem=267G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

EPOCHS=$(seq 0 39)
i=0
for train_dir in train_VGG16_synthdata_tree_nclass_50_n_exm_1000 ; do
      for epoch in ${EPOCHS[@]} ; do
        train_dir_list[$i]="$train_dir"
        epoch_list[$i]="$epoch"
        i=$i+1
  done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running training  ${train_dir_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running epoch ${epoch_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds.simg python ~/MyCodes/neural_manifolds/run_mftma_on_epoch_data.py "${train_dir_list[$SLURM_ARRAY_TASK_ID]}" "${epoch_list[$SLURM_ARRAY_TASK_ID]}"
