res=create_synth_data_cholesky_method('structure','tree','n_class',20,'exm_per_class',2,'n_feat',2000,'beta',.4,'sigma',5,'norm',false);
%plot(res.graph.full_graph,'Layout','force')
plot_tree_decomp(res.data);
kemp_data=load('./dataset_tests/synthtree.mat');
plot_tree_decomp(kemp_data.data);


%% 
res=create_synth_data_cholesky_method('structure','tree','n_class',8,'exm_per_class',2,'n_feat',936,'beta',.4,'sigma',4,'norm',true,'save',false);
plot(res.graph.class_graph,'-b','NodeLabel',{})
figure;plot(res.graph.class_graph,'Layout','force')
plot_tree_decomp(res.data_full);
