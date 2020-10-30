#!/bin/sh
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_knn='knn-k=100-dist_metric=euclidean-num_subsamples=100'

i=0
LINE_COUNT=0

GRAND_KNN_FILE="${ANALYSIS_DIR}/Grand_pool_${analyze_knn}_processed.csv"
rm -f $GRAND_KNN_FILE
touch $GRAND_KNN_FILE

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
        EXT_FILE="master_${model}_extracted.csv"
        FULL_FILE="${ROOT_DIR}/${model}/${EXT_FILE}"
        echo $FULL_FILE
        MODEL_LINE=0
        #TODO make pkl to .mat transformation
        while read line; do
	            printf "%d, %d , %s, %s, %s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze_knn" "$line" >> $GRAND_KNN_FILE
                LINE_COUNT=$(expr ${LINE_COUNT} + 1)
                MODEL_LINE=$(expr ${MODEL_LINE} + 1)
	      done <$FULL_FILE
        i=$i+1
      done
    done
  done
done


echo $LINE_COUNT
#nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 1300 1500 200 knn_script.sh $GRAND_KNN_FILE &


