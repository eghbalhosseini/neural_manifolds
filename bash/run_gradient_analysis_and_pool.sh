#!/bin/sh
#SBATCH --job-name=grad_
#SBATCH --array=0-1
#SBATCH --time=72:00:00
#SBATCH --mem=256G
#SBATCH -N 1
#SBATCH --exclude node017,node018

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'
OVERWRITE='true'
i=0


struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.000161  ; do
  for sigma in 5.0  ; do
    for nclass in 64  ; do
      for net in NN  ; do
        for idx in 0 ; do
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

echo "Running model:  ${model_list[$SLURM_ARRAY_TASK_ID]}"


singularity exec -B /om:/om,/mindhive:/mindhive /om/user/${USER}/simg_images/neural_manifolds.simg python /om/user/${USER}/neural_manifolds/gradient_analysis.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${analyze_mftma} ${train_dir_list[$SLURM_ARRAY_TASK_ID]}  $OVERWRITE
