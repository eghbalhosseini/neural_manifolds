#!/bin/sh
#SBATCH --job-name=mftma_pool
#SBATCH --array=0-1
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --exclude node017,node018

ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
i=0
for model in NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed; do
               # combine configs
                    model_list[$i]="$model"
                    i=$[$i + 1]
done

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

echo "Running model:  ${model_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running analysis:  f$analyze"

singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/mftma_pool_results.py ${model_list[$SLURM_ARRAY_TASK_ID]} ${analyze}