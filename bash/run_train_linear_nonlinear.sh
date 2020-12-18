#!/bin/sh

#SBATCH --job-name=run_train
#SBATCH -t 8:00:00
#SBATCH --array=0-1
#SBATCH --mem=80000
#SBATCH --exclude node017,node018

# create a list of config names

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
#chmod g+w -R "${ROOT_DIR}"

struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

i=0
for beta in 0.016 ; do
  for sigma in 0.833 ; do
    for nclass in 64 ; do
      for idx in 0  ; do
        for nfeat in 936 ; do
          for network in linear_NN NN ; do

        model="${network}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=${nfeat}-train_test-fixed"
        model_list[$i]="$model"
        i=$i+1
        done
      done
    done
  done
done
done
# define singularity paths
module add openmind/singularity
SINGULARITY_CACHEDIR=/om/user/${USER}/st/
export SINGULARITY_CACHEDIR
#
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME
# run training
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running model:  ${model_list[$SLURM_ARRAY_TASK_ID]}"

singularity exec --nv -B /om:/om,/mindhive:/mindhive /om/user/${USER}/simg_images/python36_fz python /om/user/${USER}/neural_manifolds/train_network_on_synthetic_data.py "${model_list[$SLURM_ARRAY_TASK_ID]}"
#wait
# Grant access
#chmod g+w -R ${ROOT_DIR}${model_list[$SLURM_ARRAY_TASK_ID]}

#NOBATCH --gres=gpu:1 --constraint=high-capacity