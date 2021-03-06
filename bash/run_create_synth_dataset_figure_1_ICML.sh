#!/bin/sh

#  run_create_synth_dataset_with_carpets.sh
#SBATCH --job-name=synth_data_carpet
#SBATCH -t 96:00:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu 300000
#SBATCH --array=1-30
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL


echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);\
addpath(genpath('/om/user/${USER}/neural_manifolds/matlab/'));\
create_figure_1_ICML_v2_openmind($SLURM_ARRAY_TASK_ID);\
quit;"


