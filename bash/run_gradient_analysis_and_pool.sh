#!/bin/sh
#SBATCH --job-name=grad_
#SBATCH --array=0
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH --exclude node017,node018

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'
OVERWRITE='true'
i=0

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

singularity exec -B /om:/om,/mindhive:/mindhive /om/user/${USER}/simg_images/neural_manifolds.simg python /om/user/${USER}/neural_manifolds/gradient_analysis.py
