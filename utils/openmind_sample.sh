#!/bin/bash
PARAM1=$1
PARAM2=$2

SINGULARITY_IMG=/om2/user/drmiguel/vagrant/newVag/low_proj.simg #path of the container
PROJECT_PATH=your/project/path #path of the project
export SINGULARITY_BINDPATH="$SINGULARITY_IMG","$PROJECT_PATH" #export path to container
singularity exec $SINGULARITY_IMG \ 
		python3 your_script.py \#execute python script inside the container
		${PARAM1} \
		${PARAM2} 