#!/bin/sh

#SBATCH --job-name=run_train
#SBATCH -t 8:00:00
#SBATCH --array=0-19
#SBATCH --mem=80000
#SBATCH --exclude node017,node018

# create a list of config names

#ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/
#chmod g+w -R "${ROOT_DIR}"


i=0
for model in NN-partition_nclass=100_nobj=100000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             NN-partition_nclass=100_nobj=100000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-test_performance \
             NN-partition_nclass=50_nobj=50000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-tree_nclass=50_nobj=50000_nhier=3_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=100_nobj=100000_nhier=4_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
              NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
              NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    i=$i+1
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

#NOBATCH --gres=gpu:1 --constraint=high-capacity