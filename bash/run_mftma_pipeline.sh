#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
overwrite='False'
i=0
for model in NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed \
             NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.02_sigma=2.5_nfeat=3072-train_test-fixed  ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"
                    echo "Running analysis:  $analyze"
                    bash mftma_analysis_pipeline.sh $model $analyze $overwrite
                    wait
                    i=$i+1
done
i=0
for model in ${model_list[@]} ; do
    chmod g+w -R ${ROOT_DIR}$model
    i=$i+1
done

