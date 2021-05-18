clear all; 
%load('0987_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=10-batchidx=510_layer_2_Linear_extracted_v3.mat')
% Xd = double(activation.projection_results{1}.layer_2_Linear); 
% X = (activation.projection_results{1}.layer_2_Linear);
load('0789_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=08-batchidx=720_layer_0_Input_extracted_v3.mat');
X = (activation.projection_results{1}.layer_0_Input);
options.n_rep =10; 
options.seed0 = 1; 
options.flag_NbyM =1; 

% for ii=1:size(X,1)
for ii=1:30
    XtotT{ii} = double(squeeze(X(ii,:,1:20))); 
end

[output] = manifold_simcap_analysis(XtotT, options);