#!/bin/bash
#
#SBATCH -n 1 # one core
#SBATCH -t 1:00:00
#SBATCH --mem=8000


OVERWRITE='false'
#
# this is where we put those lines ,

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec -B /om:/om,/mindhive:/mindhive /om/user/`whoami`/simg_images/neural_manifolds_tiny.simg python /om/user/`whoami`/neural_manifolds/extract_data_mock.py