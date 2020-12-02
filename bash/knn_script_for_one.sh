#!/bin/bash
#
#SBATCH -c 1
#SBATCH --exclude node[017-018]
#SBATCH -t 3:00:00


#GRAND_FILE=$1
#MODEL_ID=$2
#ANALYZE_ID=$3
#OVERWRITE='false'
#ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/

#echo "model ${model}"
#echo "analyze ${run_analyze}"
#echo "layer to analyze ${run_layer}"

module add mit/matlab/2020a
matlab -nodisplay -r "addpath(genpath('/om/user/`whoami`/neural_manifolds/'));\
runKNN('root_dir','/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/','analyze_identifier','knn-k=100-dist_metric=euclidean-num_subsamples=100','model_identifier','NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed','layer','layer_3_Linear','dist_metric','euclidean','k',100,'num_subsamples',100);quit;"

#runKNN('root_dir','$ROOT_DIR','analyze_identifier','$run_analyze','model_identifier','$run_model','layer','$run_layer','dist_metric','$run_dist_metric','k',$run_k,'num_subsamples',$run_num_subsamples);\
#fprintf('done');quit;"

#Running one example test:
# runKNN('root_dir','/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/','analyze_identifier','knn-k=100-dist_metric=euclidean-num_subsamples=100','model_identifier','NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed','layer','0_Input','dist_metric','euclidean','k',100,'num_subsamples',100)

