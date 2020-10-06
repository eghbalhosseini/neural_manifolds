#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/om/group/evlab/Greta_Eghbal_manifolds/analyze/
analyze='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
overwrite='False'
i=0
LINE_COUNT=0
GRAND_FILE="${ANALYSIS_DIR}/Grand_pool_${analyze}.csv"
touch $GRAND_FILE
for model in NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"
                    echo "Running analysis:  $analyze"
                    EXT_FILE="master_${model}_extracted.csv"
                    FULL_FILE="${ROOT_DIR}/${model}/${EXT_FILE}"
                    echo $FULL_FILE
                    while read line; do

	                    all_analysis_files[$LINE_COUNT]=$line
                      LINE_COUNT=$(expr ${LINE_COUNT} + 1)
	                    printf "%d,%s,%s,%s\n" "$LINE_COUNT" "$model" "$analyze" "$line" >> $GRAND_FILE

	                  done <$FULL_FILE
	                  i=$i+1


done


nohup /cm/shared/admin/bin/submit-many-jobs LINE_COUNT 1300 1500 200 mftma_knn_script.sh GRAND_FILE


