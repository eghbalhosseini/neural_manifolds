#!/bin/sh
#SBATCH --job-name=distance_pool
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --exclude node017,node018

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
i=0

struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.0 ; do
  for sigma in 0.5 ; do
    for nclass in 64 ; do
      for idx in 0 ; do
        for net in NN ; do
        model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
        model_list[$i]="$model"
        i=$i+1
        done
      done
    done
  done
done


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

echo "Running pooling for:  ${model_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec -B /om:/om,/mindhive:/mindhive /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/distance_pool_results.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${analyze_mftma}
