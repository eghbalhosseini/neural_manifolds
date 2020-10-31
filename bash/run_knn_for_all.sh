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

# get distance metric
dist_metric=$(echo "$analyze_knn" | grep -a -o "dist_metric=.*-num")
dist_metric=${dist_metric//dist_metric=/}
dist_metric=${dist_metric//-num/}

# get k
k=$(echo "$analyze_knn" | grep -a -o "k=.*-dist")
k=${k//k=/}
k=${k//-dist/}

# get num_subsamples
num_subsamples=$(echo "$analyze_knn" | grep -a -o "num_subsamples=.*")
num_subsamples=${num_subsamples//num_subsamples=/}

for beta in 0.0 0.016 0.033 0.05 ; do
  for sigma in 0.0 0.833 1.667 2.5 ; do
    for nclass in 64 96 ; do
      for idx in 0 1 ; do
        model="NN-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=3072-train_test-fixed"
        model_list[$i]="$model"
        EXT_FILE="master_${model}_extracted.csv"
        FULL_FILE="${ROOT_DIR}/${model}/${EXT_FILE}"
        echo $FULL_FILE
        if [ -f "$FULL_FILE" ] ; then
          # first create an auxilary with fix nulls
          AUX_FILE="master_${model}_extracted_aux.csv"
          FULL_AUX_FILE="${ROOT_DIR}/${model}/${AUX_FILE}"
          tr < $FULL_FILE '\000' '\n' > $FULL_AUX_FILE

        # get layers :
          Layers=$(grep -a -o "layer.*extracted" $FULL_AUX_FILE | sort -u)
          Layers=${Layers//_extracted/}
          MODEL_LINE=0

          for layer in ${Layers[@]} ; do
            printf "%d,%d,%s,%s,%s,%s,%s,%s\n" "$LINE_COUNT" "$MODEL_LINE" "$model" "$analyze_knn" "$layer" "$k" "$dist_metric" "$num_subsamples" >> $GRAND_KNN_FILE
            LINE_COUNT=$(expr ${LINE_COUNT} + 1)
            MODEL_LINE=$(expr ${MODEL_LINE} + 1)
          done
          rm -f $FULL_AUX_FILE
        else
          echo "$FULL_FILE doesnt exist yet."
        fi
        i=$i+1
      done
    done
  done
done

chmod g+w -R /mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze
echo $LINE_COUNT
nohup /cm/shared/admin/bin/submit-many-jobs 3 2 3 1 knn_script.sh $GRAND_KNN_FILE &
#nohup /cm/shared/admin/bin/submit-many-jobs $LINE_COUNT 1300 1500 200 knn_script.sh $GRAND_KNN_FILE &


