#!/bin/sh
#SBATCH --job-name=mftma_epoch
#SBATCH --output=mftma_epoch_%j.out
#SBATCH --error=mftma_epoch_%j.err
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH -t 04:00:00
#SBATCH -c 1
#SBATCH --exclude node[017-092]

# echo "Running training  ${1}"
# echo "Running epoch ${2}"

timestamp() {
  date +"%T"
}

filename="mftma_epoch_"$(date '+%Y%m%d%T')".txt"

# cd /om/user/`whoami`/neural_manifolds/

# python run_mftma_on_epoch_data_noargs.py > "$filename"

module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
export RESULTCACHING_HOME
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds.simg python /om/user/gretatu/neural_manifolds/run_mftma_on_epoch_data_noargs.py 

#python run_mftma_on_epoch_data.py "${1}" "${2}" > "$filename"
