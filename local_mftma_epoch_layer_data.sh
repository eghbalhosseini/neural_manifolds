#!/bin/zsh

LAYERS=$(seq 0 16)
i=0
for n_class in 50 ; do
  for exm_per_class in 100 ; do
    for data in synth_tree_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat \
      synth_partition_nobj_50000_nclass_50_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat \
      synth_partition_nobj_100000_nclass_100_nfeat_3072_beta_0.01_sigma_1.50_norm_1.mat  ; do
      for layer in ${LAYERS[@]} ; do
        dataset_list[$i]="$data"
        layer_list[$i]="$layer"
        n_class_list[$i]="$n_class"
        exm_per_class_list[$i]="$exm_per_class"
        i=$i+1
      done
    done
  done
done

lines=$i
echo $lines