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


load(strcat('/Users/eghbalhosseini/Desktop/mftma_capacity_data/',...
    '0000_NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.000161_sigma=5.0_nfeat=936-train_test-fixed-epoch=01-batchidx=15_layer_0_Input_extracted_v3.mat'));
output_cell={};
for p=1:size(activation.projection_results,2) 
X = (activation.projection_results{p}.(activation.layer_name));
options.n_rep =10;
options.seed0 = 1;
options.flag_NbyM =1;
XtotT={};
for ii=1:size(X,1),
    X_class=double(squeeze(X(ii,:,1:size(X,3))));
    modif=0e-2*repmat(randn(size(X_class,1),1),1,size(X_class,2));
    XtotT{ii} = X_class;%+modif;
end
[output] = manifold_simcap_analysis(XtotT, options);

%save(strcat(save_path,filesep,mftma_id,filesep,network_id,filesep,train_id,filesep,sav_id),'output');

output_cell=[output_cell;output]
end 

mftma=[ 0.304488, 0.305600, 0.281324, 0.223159];

sims=cellfun(@(x) x.asim0, output_cell)
classes=[64,32,16,8]
colors=flipud(inferno(6));
figure
axes=subplot(1,1,1)
hold on
arrayfun(@(x) scatter(sims(x),mftma(x),20,colors(x+2,:),'filled','displayname',num2str(classes(x))),[1:4])
%scatter(sims(1:4),mftma,20,colors(3:6,:),'filled')
xlim([0,.5])
ylim([0,.5])
axis square
hold on 
plot([0,.5],[0,.5],'--')
xlabel('simulation')
ylabel('capacity')
shg
legend('show')
title('input,  936 feature, epoch 1')


