#!/bin/bash

#SBATCH --job-name=pMAP
#SBATCH --array=0-18
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --constraint="pascal|turing|volta"
#SBATCH --mail-type=ALL
#SBATCH --exclude node017,node018
#SBATCH --mail-user=ehoseini@mit.edu


model_identifier="NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed"
train_identifier="epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06"
analyze_identifier="mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5"


i=0
for layer in layer_1_Linear layer_2_Linear layer_3_Linear ; do
  for hier in 0 1 2 3 4 5 ; do
        layer_list[$i]="$layer"
        hier_list[$i]="$hier"
        i=$i+1
  done
done


module add openmind/singularity
export SINGULARITY_CACHEDIR=/om/user/${USER}/st/
XDG_CACHE_HOME=/om/user/${USER}/st
export XDG_CACHE_HOME

# Get the relevant line from the parameters
singularity exec --nv -B /om:/om,/mindhive:/mindhive,/om2:/om2 /om/user/${USER}/simg_images/neural_manifolds_cuda_p_map /om/user/${USER}/neural_manifolds/bash/p_map_permission.sh ${model_identifier} ${train_identifier} ${analyze_identifier} ${layer_list[$SLURM_ARRAY_TASK_ID]} ${hier_list[$SLURM_ARRAY_TASK_ID]}

