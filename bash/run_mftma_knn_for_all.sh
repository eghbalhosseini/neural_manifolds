#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/om/group/evlab/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
analyze_knn='knn-k=100-dist_metric=euclidean-num_subsamples=100'
overwrite='False'
i=0
LINE_COUNT=0
GRAND_MFTMA_FILE="${ANALYSIS_DIR}/Grand_pool_${analyze}.csv"
GRAND_KNN_FILE="${ANALYSIS_DIR}/Grand_pool_${analyze}.csv"
touch $GRAND_MFTMA_FILE
touch $GRAND_KNN_FILE
for model in NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                model_list[$i]="$model"
                echo "Running model:  ${model_list[$i]}"
                EXT_FILE="master_${model}_extracted.csv"
                FULL_FILE="${ROOT_DIR}/${model}/${EXT_FILE}"
                echo $FULL_FILE
                while read line; do
	                    printf "%d, %s, %s, %s\n" "$LINE_COUNT" "$model" "$analyze_mftma" "$line" >> $GRAND_MFTMA_FILE
	                    printf "%d, %s, %s, %s\n" "$LINE_COUNT" "$model" "$analyze_knn" "$line" >> $GRAND_KNN_FILE
                      LINE_COUNT=$(expr ${LINE_COUNT} + 1)
	              done <$FULL_FILE
	              i=$i+1
done

#nohup /cm/shared/admin/bin/submit-many-jobs LINE_COUNT 1300 1500 200 mftma_script.sh $GRAND_MFTMA_FILE
nohup /cm/shared/admin/bin/submit-many-jobs LINE_COUNT 1300 1500 200 knn_script.sh $GRAND_KNN_FILE


