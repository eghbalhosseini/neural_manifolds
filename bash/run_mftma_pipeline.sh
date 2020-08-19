#!/bin/sh

i=0
for model in NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
            NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"
                    bash mftma_analysis_pipeline.sh $model
                    i=$i+1
done
i=0
for model in ${model_list[@]} ; do
    echo "pooling model:  $model"
    bash mftma_pool.sh $model
    i=$i+1
done
wait
