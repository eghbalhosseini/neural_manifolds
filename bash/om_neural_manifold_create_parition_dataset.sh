#!/bin/sh

#  om_neural_manifold_create_partiion_dataset.sh

#SBATCH --job-name=PARTITION
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem-per-cpu 32000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --output=PARTITION_result_%j.out
#SBATCH --error=PARTITION_result_%j.err


module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);addpath('/home/ehoseini/MyCodes/neural_manifolds/');\
addpath(genpath('/home/ehoseini/MyCodes/neural_manifolds/'));\
save_path='/om/user/ehoseini/MyData/neural_manifolds/';\
n_class=50;exm_per_class=1000;n_feat=3*32*32;\
beta=0.01;sigma=1.5;\
neural_manifold_create_partition_dataset_cholesky_method('n_class',n_class,'exm_per_class',exm_per_class,'n_feat',n_feat,'save_path',save_path,'beta',beta,'sigma',sigma);\
quit;"