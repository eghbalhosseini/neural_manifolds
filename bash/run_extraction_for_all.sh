#!/bin/sh
ROOT_DIR=/om/group/evlab/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'
overwrite='False'
i=0
LINE_COUNT=0
GRAND_FILE="${ROOT_DIR}/Grand_pool_${analyze}_extracted.csv"
touch $GRAND_FILE

struct_list="partition tree"
hier_list="1 6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.0 0.016 0.033 0.05 ; do
  for sigma in 0.0 0.833 1.667 2.5 ; do
    for nclass in 64 96 ; do
      for idx in 0 1 ; do
        model="NN-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=3072-train_test-fixed"
        model_list[$i]="$model"
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
    done
  done
done


echo $LINE_COUNT
nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 1300 1500 200 new_extraction_script.sh $GRAND_FILE


