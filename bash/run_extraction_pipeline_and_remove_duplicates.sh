#!/bin/sh
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted
analyze='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'
EXTERACTION_DIR="${ROOT_DIR}/${analyze}/"
if [ -d "$EXTERACTION_DIR" ]
              then
                true
              else
    mkdir $EXTERACTION_DIR
fi

i=0
LINE_COUNT=0
GRAND_FILE="${ROOT_DIR}/Grand_pool_${analyze}_extracted.csv"
rm -f $GRAND_FILE
touch $GRAND_FILE

struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.000161 ; do
  for sigma in 5.0  ; do
    for nclass in 64 ; do
      for idx in 0 ; do
        for net in NN linear_NN ; do
          for train_dir in epochs-10_batch-32_lr-0.001_momentum-0.5_init-gaussian_std-0.0001 \
                           epochs-10_batch-32_lr-0.002_momentum-0.6_init-gaussian_std-1e-05 \
                           epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06 ; do
        model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
        model_list[$i]="$model"
        PTH_FILE="master_${model}.csv"
        FULL_FILE="${ROOT_DIR}/${model}/${train_dir}/${PTH_FILE}"
        echo $FULL_FILE
        MODEL_LINE=0
        while read line; do
              original="${ROOT_DIR}/${model}/${train_dir}/"
              file_name="${line/$original/}"
              original='.pth'
              correction='_layer*'
              possible_file="${file_name/$original/$correction}"
              leading_zero=$(printf "%04d" $MODEL_LINE)
              new_file_name="${ROOT_DIR}/${analyze}/${model}/${train_dir}/${leading_zero}_${possible_file}"
              #if [ -f "$new_file_name" ]
              if compgen -G "${new_file_name}" > /dev/null ; then
                #echo "$new_file_name exists"
                true
              else
                LINE_COUNT=$(expr ${LINE_COUNT} + 1)
                echo "$new_file_name doesnt exists adding it"
	              printf "%d, %d , %s, %s, %s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze" "$line" >> $GRAND_FILE

              fi
              MODEL_LINE=$(expr ${MODEL_LINE} + 1)
	      done <$FULL_FILE

        i=$i+1
        done
        done
      done
    done
  done
done

echo $LINE_COUNT
run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
    nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 800 1000 200 extraction_script_for_all_permission.sh $GRAND_FILE &
  else
    echo $LINE_COUNT
fi
#nohup /cm/shared/admin/bin/submit-many-jobs 20 15 20 5 extraction_script_for_all_permission.sh $GRAND_FILE &


