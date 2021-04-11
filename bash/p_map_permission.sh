#!/bin/bash

model_id=$1
train_id=$2
analyze_id=$3
layer_id=$4
hier_id=$5

source /opt/conda/bin/activate

#conda activate rapids

#conda list

#conda env list
#conda deactivate rapids
#conda activate base
which python

python /om/user/${USER}/neural_manifolds/poincare_map_analysis.py ${model_id} ${train_id} ${analyze_id} ${layer_id} ${hier_id}