#!/bin/sh
#SBATCH --job-name=Pool
#SBATCH -t 2:00:00
#SBATCH -N 1
#SBATCH --array=0
#SBATCH --exclude node017,node018
#SBATCH --mail-type=ALL

ROOT_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/
ANALYSIS_DIR=/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/analyze/
analyze_mftma='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'
i=0
struct_list="tree"
hier_list="6"
struct_arr=($struct_list)
hier_arr=($hier_list)

for beta in 0.000161 ; do
    for sigma in 5.0  ; do
      for nclass in 64  ; do
        for net in NN ; do
          for idx in 0 ; do
            for train_dir in epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06 ; do
            model="${net}-${struct_arr[$idx]}_nclass=${nclass}_nobj=$(($nclass * 1000))_nhier=${hier_arr[$idx]}_beta=${beta}_sigma=${sigma}_nfeat=936-train_test-fixed"
            model_list[$i]="$model"
            train_dir_list[$i]="$train_dir"
            i=$i+1
          done
        done
      done
    done
  done
done
# implement the code in matlab
module add mit/matlab/2020a
matlab -nodisplay -r "maxNumCompThreads($SLURM_NTASKS);\
addpath(genpath('/om/user/${USER}/neural_manifolds/matlab/'));\
loc_folder=strcat('${ROOT_DIR}',filesep,'${analyze_mftma}',filesep,'${model_list[$SLURM_ARRAY_TASK_ID]}',filesep,'${train_dir_list[$SLURM_ARRAY_TASK_ID]}');\
disp(loc_folder);\
model_identifier='${model_list[$SLURM_ARRAY_TASK_ID]}';\
d=dir(strcat(loc_folder,filesep,'*_capacity_v3.mat'));\
names=arrayfun(@(x) d(x).name,1:size(d,1),'uni',false);\
file_num=regexp(names,['\d+_',model_identifier],'match');\
disp(model_identifier);\
layer_num=regexp(names,'layer_\w*_capacity','match');\
layer_num=cellfun(@(x) cell2mat(erase(x,'_capacity')),layer_num,'uni',false);\
file_num_unorder=cell2mat(cellfun(@(x) str2num(cell2mat(erase(x,['_',model_identifier]))),file_num,'uni',false));\
[~,order_idx]=sort(file_num_unorder);\
disp('test');\
sorted_files=arrayfun(@(x) strcat(d(x).folder,filesep,d(x).name),order_idx,'uni',false);\
capacity_data_pool=[];\
for k=1:length(sorted_files),capacity_data_pool=[capacity_data_pool;[file_num_unorder(k),layer_num{k},{load(sorted_files{k})}]];end;\
save_path=strcat(loc_folder,filesep,model_identifier,'_capacity_pooled.mat');\
save(save_path,'capacity_data_pool');\
quit;"
