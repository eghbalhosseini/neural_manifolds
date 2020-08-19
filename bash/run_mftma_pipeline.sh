#!/bin/sh

i=0
for model in NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
            NN-tree_nclass=50_nobj=50000_nhier=3_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed ; do

               #NN-partition_nclass=100_nobj=100000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-test_performance \
             #NN-partition_nclass=50_nobj=50000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=64_nobj=64000_nhier=1_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=50_nobj=50000_nhier=3_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=100_nobj=100000_nhier=4_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             # NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
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
