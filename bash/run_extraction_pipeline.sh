#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/

i=0
for model in NN-partition_nclass=96_nobj=96000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"
                    bash extraction_pipeline.sh $model
                    wait
                    i=$i+1
done

wait
# Grant access
chmod g+w -R ${ROOT_DIR}${model_list[$SLURM_ARRAY_TASK_ID]}