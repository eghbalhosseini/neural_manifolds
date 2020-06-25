#!/bin/sh
#SBATCH --job-name=mftma_extr
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu

i=0
for n_class in 50 ; do
  for exm_per_class in 100 ; do
    for data in synth_tree_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat \
      synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat \
      synth_partition_nobj_100000_nclass_100_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat  ; do
        dataset_list[$i]="$data"
        n_class_list[$i]="$n_class"
        exm_per_class_list[$i]="$exm_per_class"
        i=$i+1
    done
  done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running dataset  ${dataset_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running n class  ${n_class_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running exm per class ${exm_per_class_list[$SLURM_ARRAY_TASK_ID]}"
module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds.simg python ~/MyCodes/neural_manifolds/extract_net_activation_from_synthetic_data.py "${dataset_list[$SLURM_ARRAY_TASK_ID]}" "${n_class_list[$SLURM_ARRAY_TASK_ID]}" "${exm_per_class_list[$SLURM_ARRAY_TASK_ID]}"
