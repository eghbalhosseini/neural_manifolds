#!/bin/bash
#SBATCH --job-name=rapids
#SBATCH --time=01:00:00
#SBACTH --ntasks=1
#SBATCH --output="rapids-%j.out"
#SBATCH --mem=10G
#SBATCH --gres=gpu:1             # 1 GPU
#SBATCH --constraint=any-gpu     # Any GPU on the cluster.

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

source /opt/conda/bin/activate
conda env list
conda activate rapids



singularity exec -B /om:/om,/mindhive:/mindhive /om/user/ehoseini/simg_images/neural_manifolds_cuda.simg python /om/user/`whoami`/neural_manifolds/test_rapids.py