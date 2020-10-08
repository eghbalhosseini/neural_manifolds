#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
overwrite='False'
i=0
LINE_COUNT=0
GRAND_FILE="${ROOT_DIR}/Grand_pool_${analyze}_extracted.csv"
touch $GRAND_FILE
for model in NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed \
             NN-tree_nclass=96_nobj=96000_nhier=6_beta=0.02_sigma=0.83_nfeat=3072-train_test-fixed ; do
               # combine configs
                    model_list[$i]="$model"
                    echo "Running model:  ${model_list[$i]}"

                    PTH_FILE="master_${model}.csv"
                    FULL_FILE="${ROOT_DIR}/${model}/${PTH_FILE}"
                    echo $FULL_FILE
                    MODEL_LINE=0
                    while read line; do
	                    printf "%d, %d , %s, %s, %s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze" "$line" >> $GRAND_FILE
                      LINE_COUNT=$(expr ${LINE_COUNT} + 1)
                      MODEL_LINE=$(expr ${MODEL_LINE} + 1)
	                  done <$FULL_FILE
	                  i=$i+1


done


#nohup /cm/shared/admin/bin/submit-many-jobs LINE_COUNT 1300 1500 200 new_extraction_script.sh $GRAND_FILE


