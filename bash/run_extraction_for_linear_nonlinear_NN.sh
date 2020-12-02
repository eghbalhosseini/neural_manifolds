#!/bin/sh
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
analyze='mftma-exm_per_class=100-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'

i=0
LINE_COUNT=0
GRAND_FILE="${ROOT_DIR}/Grand_pool_${analyze}_extracted.csv"
rm -f $GRAND_FILE
touch $GRAND_FILE

struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.016 ; do
  for sigma in 0.833  ; do
    for nclass in 64 ; do
      for idx in 0 ; do
        for net in NN ; do
        model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
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
done


echo $LINE_COUNT
nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 200 300 100 extraction_script.sh $GRAND_FILE &


