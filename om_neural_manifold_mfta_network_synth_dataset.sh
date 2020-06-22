#!/bin/sh
#SBATCH --job-name=mftma_vgg
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=126G
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --output=MFTMA_result_%j.out
#SBATCH --error=MFTMA_result_%j.err

i=0
for data in synth_tree_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat \
  synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat  ; do
    dataset_list[$i]="$data"
    i=$i+1
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running dataset  ${dataset_list[$SLURM_ARRAY_TASK_ID]}"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/python36.simg python ~/MyCodes/neural_manifolds/run_mftma_on_synthetic_data.py "${dataset_list[$SLURM_ARRAY_TASK_ID]}"
