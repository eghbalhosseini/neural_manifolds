#!/bin/sh

#  om_neural_manifold_create_partiion_dataset.sh
#SBATCH --output=partition_data.out
#SBATCH --job-name=PARTITION
#SBATCH -t 18:00:00
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --mem-per-cpu 10000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);addpath('/home/ehoseini/MyCodes/neural_manifolds/');addpath(genpath('/home/ehoseini/MyCodes/neural_manifolds/'));neural_manifold_create_partition_dataset;quit;"
