#!/bin/sh
#SBATCH --job-name=mftma_epoch
#SBATCH --array=0-10
#SBATCH -t 5000
#SBATCH -N 1 # on one node
#SBATCH -n 1 # one core
#SBATCH --partition=normal
#SBATCH --mem=10G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

LAYERS=$(seq 0 16)
EPOCHS=$(seq 1 15)
i=0
for train_dir in train_VGG16_synthdata_tree_nclass_50_n_exm do
  for epoch in ${EPOCHS[@]} ; do
      for layer in ${LAYERS[@]} ; do
        train_dir_list[$i]="$train_dir"
        layer_list[$i]="$layer"
        epoch_list[$i]="$epoch"
        i=$i+1
      done
    done
done

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds.simg python ~/MyCodes/neural_manifolds/run_mftma_on_layer_epoch_data.py "${train_dir_list[$SLURM_ARRAY_TASK_ID]}" "${epoch_list[$SLURM_ARRAY_TASK_ID]}" "${layer_list[$SLURM_ARRAY_TASK_ID]}"
