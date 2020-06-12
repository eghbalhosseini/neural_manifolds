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


module add mit/matlab/2019b
matlab -nodisplay -signelCompThread -r "addpath('/home/ehoseini/MyCodes/CSGproject/');addpath(genpath('/home/ehoseini/MyCodes/CSGproject/'));addpath(genpath('/home/ehoseini/MyCodes/CSGproject/'));[f f_scaled f_scaled_trunc g]=gptest;data_folder='/home/ehoseini/MyData/CSGProject/GPFSimulation/';mkdir(data_folder);cd(data_folder);save('outputdata','f','f_scaled','g');quit;"
