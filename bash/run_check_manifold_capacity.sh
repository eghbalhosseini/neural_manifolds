#!/bin/sh

#  run_create_synth_dataset_cholesky_method.sh

#SBATCH --job-name=SIM_CAP
#SBATCH -t 48:00:00
#SBATCH -N 1
#SBATCH --mem-per-cpu 64000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL


module add mit/matlab/2018b
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);\
addpath(genpath('/om/user/${USER}/neural_manifolds/matlab/'));\
save_path='/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/data/';\
check_simulation_capacity;\
quit;"
chmod g+w -R /nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/data
