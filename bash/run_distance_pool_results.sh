#!/bin/sh
#SBATCH --job-name=distance_pool
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH -N 1
#SBATCH --exclude node017,node018

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'
i=0

struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.000161 0.0923671 ; do
  for sigma in 5.0 ; do
    for nclass in 64 ; do
      for idx in 0 ; do
        for net in NN linear_NN ; do
          for train_dir in epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06 ; do
            model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
        model_list[$i]="$model"
        train_dir_list[$i]="$train_dir"
        i=$i+1
        done
        done
      done
    done
  done
done


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

echo "Running pooling for:  ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "looking at dir for:  ${train_dir_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec -B /om:/om,/mindhive:/mindhive /om/user/${USER}/simg_images/neural_manifolds_tiny_fz.simg python /om/user/${USER}/neural_manifolds/distance_pool_results.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${analyze_mftma} ${train_dir_list[$SLURM_ARRAY_TASK_ID]}

#### USE ARRAY IF SEVERAL ####
#SBATCH --array=0-2
