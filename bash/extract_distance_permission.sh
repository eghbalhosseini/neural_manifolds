#!/bin/bash

run_file=$1
run_model_line=$2
run_model=$3
run_analyze=$4
OVERWRITE=$5

source ~/.bashrc
python /om/user/${USER}/neural_manifolds/extract_data_mock.py ${run_file} ${run_model_line} ${run_model} ${run_analyze} ${OVERWRITE}