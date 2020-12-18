#!/bin/sh

#  run_create_synth_dataset_cholesky_method.sh

#SBATCH --job-name=SYNTH
#SBATCH -t 1:00:00
#SBATCH -N 1
#SBATCH --array=0-1
#SBATCH -n 4
#SBATCH --mem-per-cpu 32000
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL


i=0
for beta_id in 1 ; do
  for sigma_id in 1 ; do
    for struct_id in 1 2 ; do
      for n_class in 64 ; do
        struct_list[$i]=$struct_id
        beta_list[$i]=$beta_id
        sigma_list[$i]=$sigma_id
        n_class_list[$i]=$n_class
        i=$i+1
      done
    done
  done
done

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Running n_class ${n_class_list[$SLURM_ARRAY_TASK_ID]}"
echo "Running structure ${struct_list[$SLURM_ARRAY_TASK_ID]}"

module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);\
addpath(genpath('/om/user/${USER}/neural_manifolds/matlab/'));\
save_path='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/data/';\
structures={'partition','tree'};\
betas=[1];\
sigmas=[1];\
fprintf('creating structure %s\n',structures{${struct_list[$SLURM_ARRAY_TASK_ID]}});\
struct=structures{${struct_list[$SLURM_ARRAY_TASK_ID]}};\
n_class=${n_class_list[$SLURM_ARRAY_TASK_ID]};\
exm_per_class=1000;n_feat=936;\
beta=betas(${beta_list[$SLURM_ARRAY_TASK_ID]});sigma=sigmas(${sigma_list[$SLURM_ARRAY_TASK_ID]});\
create_synth_data_cholesky_method('structure',struct,'n_class',n_class,'exm_per_class',exm_per_class,'n_feat',n_feat,'save_path',save_path,'beta',beta,'sigma',sigma);\
quit;"
chmod g+w -R /mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/data
