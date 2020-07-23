#!/bin/bash


LAYERS=$(seq 0 16)
EPOCHS=$(seq 1 15)
i=0
for train_dir in train_VGG16_synthdata_tree_nclass_50_n_exm_1000 ; do
  for epoch in ${EPOCHS[@]} ; do
      for layer in ${LAYERS[@]} ; do
        train_dir_list[$i]="$train_dir"
        layer_list[$i]="$layer"
        epoch_list[$i]="$epoch"
        i=$i+1
      done
    done
done

lines=$(seq 0 254)
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate manifold


for line in ${lines[@]} ; do
 echo "running training ${train_dir_list[line]}"
  echo "Running layer ${layer_list[line]}"
  echo "Running epoch ${epoch_list[line]}"
  python /Users/eghbalhosseini/MyCodes/neural_manifolds/run_mftma_on_layer_epoch_data.py ${train_dir_list[line]} ${epoch_list[line]} ${layer_list[line]}
done