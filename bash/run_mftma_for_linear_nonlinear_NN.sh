#!/bin/sh
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=False-kappa=0-n_t=300-n_rep=1'

i=0
LINE_COUNT=0
GRAND_MFTMA_FILE="${ANALYSIS_DIR}/Grand_pool_${analyze_mftma}_processed.csv"
rm -f $GRAND_MFTMA_FILE
touch $GRAND_MFTMA_FILE
#
#struct_list="partition tree"
#hier_list="1 6"
struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)
old="extracted.pkl"
new="mftma_analysis.pkl"
for beta in 0.016 ; do
  for sigma in 0.833 ; do
    for nclass in 64 ; do
      for net in NN  ; do
        for idx in 0 ; do
          model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
          model_list[$i]="$model"
          EXT_FILE="master_${model}_extracted.csv"
          FULL_FILE="${ROOT_DIR}/${model}/${EXT_FILE}"
          echo $FULL_FILE
          MODEL_LINE=0
          while read line; do
            # TODO : make the script check whether the file exists before adding it to the queue
               # mftma_file=$line

             # new_line="${mftma_file/$old/$new}"
	              printf "%d, %d , %s, %s, %s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze_mftma" "$line" >> $GRAND_MFTMA_FILE

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
nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 1000 1200 200 mftma_script.sh $GRAND_MFTMA_FILE &


