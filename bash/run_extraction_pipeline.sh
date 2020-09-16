#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
i=0
for model in NN-partition_nclass=100_nobj=100000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed \
             NN-partition_nclass=50_nobj=50000_nhier=1_beta=0.0_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"
                    echo "Running analysis:  $analyze"
                    bash extraction_pipeline.sh $model $analyze
                    wait
                    i=$i+1
done
# Grant access
chmod g+w -R ${ROOT_DIR}${model_list[$SLURM_ARRAY_TASK_ID]}