#!/bin/sh
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/

i=0
LINE_COUNT=0


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

        AUX_PTH_FILE="master_${model}_aux.csv"
        AUX_FILE="${ROOT_DIR}/${model}/${AUX_PTH_FILE}"
        rm -f $AUX_FILE
        touch $AUX_FILE
        echo $FULL_FILE
        MODEL_LINE=0
        original="/om/group/evlab/Greta_Eghbal_manifolds"
        correction="/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds"

        while read line; do
              new_line="${line/$original/$correction}"
              echo "${line/$original/$correction}"
              printf "%s\n" "$new_line" >> $AUX_FILE

                LINE_COUNT=$(expr ${LINE_COUNT} + 1)
                MODEL_LINE=$(expr ${MODEL_LINE} + 1)
	      done <$FULL_FILE
        mov $AUX_file $FULL_FILE
        i=$i+1
      done
    done
  done
done

