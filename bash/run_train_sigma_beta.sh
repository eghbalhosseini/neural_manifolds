#!/bin/sh

#SBATCH --job-name=run_train
#SBATCH -t 8:00:00
#SBATCH --array=0-63
#SBATCH --mem=80000
#SBATCH --exclude node017,node018

# create a list of config names

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
#chmod g+w -R "${ROOT_DIR}"

struct_list="partition tree"
hier_list="1 6"
struct_arr=($struct_list)
hier_arr=($hier_list)

i=0
for beta in 0.0 0.016 0.033 0.05 ; do
  for sigma in 0.0 0.833 1.667 2.5 ; do
    for nclass in 64 96 ; do
      for idx in 0 1 ; do

        model="NN-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=3072-train_test-fixed"
        model_list[$i]="$model"
        i=$i+1
      done
    done
  done
done
# define singularity paths
module add openmind/singularity
# TODO : check whether this is correct
SINGULARITY_CACHEDIR=/om/user/`whoami`/st/
export SINGULARITY_CACHEDIR
#
XDG_CACHE_HOME=/om/user/`whoami`/st
export XDG_CACHE_HOME
# run training
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model:  ${model_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec --nv -B /om:/om /om/user/`whoami`/simg_images/python36_fz python /om/user/`whoami`/neural_manifolds/train_network_on_synthetic_data.py "${model_list[$SLURM_ARRAY_TASK_ID]}"
#wait
# Grant access
#chmod g+w -R ${ROOT_DIR}${model_list[$SLURM_ARRAY_TASK_ID]}

#NOBATCH --gres=gpu:1 --constraint=high-capacity