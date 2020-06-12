#!/bin/sh

#  om_neural_manifold_create_partiion_dataset.sh
#SBATCH --output=partition_data.out
#SBATCH --job-name=PARTITION
#SBATCH -t 12:00:00
#SBATCH -n 12
#SBATCH --mem-per-cpu 10000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu


module add mit/matlab/2020a
matlab -nodisplay -signelCompThread -r "addpath('/home/ehoseini/MyCodes/neural_manifolds/');addpath(genpath('/home/ehoseini/MyCodes/neural_manifolds/'));neural_manifold_create_partiion_dataset;quit;"
