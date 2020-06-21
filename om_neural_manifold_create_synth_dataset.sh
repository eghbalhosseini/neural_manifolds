#!/bin/sh

#  om_neural_manifold_create_synth_dataset.sh

#SBATCH --job-name=SYNTH
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --array=0-3
#SBATCH -n 8
#SBATCH --mem-per-cpu 32000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ehoseini@mit.edu
#SBATCH --output=SYNTH_result_%j.out
#SBATCH --error=SYNTH_result_%j.err

i=0
for struct_id in 1 2 ; do
for n_class in 50 100 ; do
  struct_list[$i]=$struct_id
  n_class_list[$i]=$n_class
  i=$i+1
done
done

echo "My SLURM_ARRAY_TASK_ID: "$SLURM_ARRAY_TASK_ID"
echo "Running structure ${struct_list[$SLURM_ARRAY_TASK_ID]}
echo "Running n_class ${n_class_list[$SLURM_ARRAY_TASK_ID]}

module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);addpath('/home/ehoseini/MyCodes/neural_manifolds/');\
addpath(genpath('/home/ehoseini/MyCodes/neural_manifolds/'));\
save_path='/om/user/ehoseini/MyData/neural_manifolds/synthetic_datasets/';\
structures={'partition','tree'};\
fprintf('creating structure %s\n',structures{$SLURM_ARRAY_TASK_ID});\
struct=structures{$struct_list[$SLURM_ARRAY_TASK_ID]};\
n_class=$n_class_list[$SLURM_ARRAY_TASK_ID];\
exm_per_class=1000;n_feat=3*32*32;\
beta=0.01;sigma=1.5;\
neural_manifold_create_synth_data_cholesky_method('structure',struct,'n_class',n_class,'exm_per_class',exm_per_class,'n_feat',n_feat,'save_path',save_path,'beta',beta,'sigma',sigma);\
quit;"
