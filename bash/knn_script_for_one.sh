#!/bin/bash
#
#SBATCH -c 1
#SBATCH --exclude node[017-018]
#SBATCH -t 05:00:00
#SBATCH --mem=10G

MODEL_ID=NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed
ANALYZE_ID=knn-k=100-dist_metric=euclidean-num_subsamples=100
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/
DIST_METRIC=euclidean
NUM_K=100
NUM_SUBSAMPLES=100

#echo "model ${model}"
#echo "analyze ${run_analyze}"
#echo "layer to analyze ${run_layer}"

# Specify a model/analyze identifier and loop through layers:
#for l={'layer_1_Linear', 'layer_2_Linear', 'layer_3_Linear'};\
module add mit/matlab/2020a
matlab -nodisplay -r "addpath(genpath('/om/user/${USER}/neural_manifolds/'));\
for l={'layer_2_Linear', 'layer_3_Linear'};\
runKNN('root_dir','$ROOT_DIR','analyze_identifier','$ANALYZE_ID','model_identifier','$MODEL_ID','layer',l{1},'dist_metric','$DIST_METRIC','k',$NUM_K,'num_subsamples',$NUM_SUBSAMPLES);\
end;\
quit;"
#runKNN('root_dir','$ROOT_DIR','analyze_identifier','$run_analyze','model_identifier','$run_model','layer','$run_layer','dist_metric','$run_dist_metric','k',$run_k,'num_subsamples',$run_num_subsamples);\
#fprintf('done');quit;"

#Running one example test:
#runKNN('root_dir','/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/','analyze_identifier','knn-k=100-dist_metric=euclidean-num_subsamples=100','model_identifier','NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.016_sigma=0.833_nfeat=936-train_test-fixed','layer','layer_2_Linear','dist_metric','euclidean','k',100,'num_subsamples',100);quit;"

