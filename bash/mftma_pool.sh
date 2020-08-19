#!/bin/sh
#SBATCH --job-name=mftma_pool
#SBATCH --array=0
#SBATCH --time=12:00:00
#SBATCH -N 1
#SBATCH --exclude node017,node018

MODEL_DIR=$1
MODEL_ID=$MODEL_DIR
ANALYZE_ID='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/mftma_pool_results.py ${MODEL_ID} ${ANALYZE_ID}