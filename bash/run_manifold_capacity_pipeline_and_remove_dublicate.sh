#!/bin/bash
ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'

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

for beta in 0.0923671 ; do
  for sigma in 5.0 ; do
    for nclass in 64 ; do
      for net in NN  ; do
        for idx in 0 ; do
          for train_dir in epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06 ; do
          model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
          FULL_DIR="${ROOT_DIR}/${analyze_mftma}/${model}/${train_dir}"
          ANALYZE_DIR="${ANALYSIS_DIR}/${analyze_mftma}/${model}/${train_dir}"
          echo "looking at ${FULL_DIR} "
          MODEL_LINE=0
          while read x; do
              # check if file already exist in analyze dir
              original='_extracted_v2.pkl'
              correction='_manifold_capacity_v2.pkl'
              possible_file="${x/$original/$correction}"
              possible_path="${possible_file/$FULL_DIR/$ANALYZE_DIR}"
              if [ -f "$possible_path" ]
              then
                true
              else
                echo "$possible_path dosent exists adding it"
                LINE_COUNT=$(expr ${LINE_COUNT} + 1)
                printf "%d, %d , %s, %s, %s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze_mftma" "$x" >> $GRAND_MFTMA_FILE

              fi
              MODEL_LINE=$(expr ${MODEL_LINE} + 1)
            done < <(find $FULL_DIR -name "*_extracted_v2.pkl")
          i=$i+1
          done
          echo $LINE_COUNT
        done
      done
    done
  done
done
echo $LINE_COUNT
run_val=0
if [ "$LINE_COUNT" -gt "$run_val" ]; then
   #nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 150 200 50 manifold_capacity_script.sh $GRAND_MFTMA_FILE &
   nohup /cm/shared/admin/bin/submit-many-jobs 20 15 20 5 mftma_manifold_capacity_scriptscript.sh $GRAND_MFTMA_FILE &
  else
    echo $LINE_COUNT
fi
#nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 1000 1200 200 manifold_capacity_script.sh $GRAND_MFTMA_FILE &
#nohup /cm/shared/admin/bin/submit-many-jobs 20 15 20 5 manifold_capacity_script.sh $GRAND_MFTMA_FILE &


