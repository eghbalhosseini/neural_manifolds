#!/bin/sh
#SBATCH --job-name=mftma_epoch
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gretatu@mit.edu

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running training  ${1}"
echo "Running epoch ${2}"

timestamp() {
  date +"%T"
}

filename="mftma_epoch_"$(date '+%Y%m%d%T')".txt"

python /om/user/`whoami`/neural_manifolds/run_mftma_on_epoch_data.py "${1}" "${2}" > "$filename"


#module add openmind/singularity
#export SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
#RESULTCACHING_HOME=/om/user/`whoami`/.result_caching
#export RESULTCACHING_HOME
#XDG_CACHE_HOME=/om/user/`whoami`/st
#export XDG_CACHE_HOME

#singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/neural_manifolds.simg python ~/MyCodes/neural_manifolds/run_mftma_on_epoch_data.py "${train_dir_list[$SLURM_ARRAY_TASK_ID]}" "${epoch_list[$SLURM_ARRAY_TASK_ID]}"
