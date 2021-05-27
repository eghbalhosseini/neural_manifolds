clear all; 
%load('0987_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=10-batchidx=510_layer_2_Linear_extracted_v3.mat')
% Xd = double(activation.projection_results{1}.layer_2_Linear); 
% X = (activation.projection_results{1}.layer_2_Linear);
save_path='/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/extracted/';
mftma_id='mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5';
network_id='NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed';
train_id='epochs-10_batch-32_lr-0.01_momentum-0.5_init-gaussian_std-1e-06';
file_id='0789_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=08-batchidx=720_layer_0_Input_extracted_v3.mat';
sav_id='0789_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=08-batchidx=720_layer_0_Input_capacity_v3.mat';
load(strcat(save_path,filesep,mftma_id,filesep,network_id,filesep,train_id,filesep,file_id));
X = (activation.projection_results{1}.layer_0_Input);
options.n_rep =10;
options.seed0 = 1;
options.flag_NbyM =1;
for ii=1:30,XtotT{ii} = double(squeeze(X(ii,:,1:20)));end
[output] = manifold_simcap_analysis(XtotT, options);
output
save(strcat(save_path,filesep,mftma_id,filesep,network_id,filesep,train_id,filesep,sav_id),'output');