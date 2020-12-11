% plot alphas and gammas for different betas, 
beta_vals=logspace(-5,2,30);
Alpha_cell={}
Gamma_cell={}
Cov_cell={}
for n=1:length(beta_vals)
res=create_synth_data_cholesky_method('structure','tree','n_class',16,'exm_per_class',20,'n_feat',936,'beta',beta_vals(n),'sigma',5,'norm',true,'save',false);
within_class_ids=cat(2,[1:length(res.class_id)],res.hierarchical_class_ids);
between_class_ids=cat(2,res.hierarchical_class_ids,[1:length(res.class_id)]*0+1);
res.data_covar=cov(res.data');

within_cell={};
between_cell={};
for i=1:length(within_class_ids)
    temp=within_class_ids{i}'*within_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    within_class=double(arrayfun(@(x,y) isequal(x,y),temp,temp1));
    within_cell{i}=within_class;
    % do the same to get between class 
    temp=between_class_ids{i}'*between_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    between_class=double(arrayfun(@(x,y) isequal(x,y),temp,temp1));
    between_cell{i}=between_class;
    
    
end 
hier_within_class={};
hier_between_class={};

for i=1:length(within_class_ids)-1
    A=(arrayfun(@(x,y) xor(x,y),within_cell{i}, within_cell{i+1}));
    hier_within_class{i}=A;
    
    B=(arrayfun(@(x,y) xor(x,y),between_cell{i}, between_cell{i+1}));
    hier_between_class{i}=B;
end 

% compute the metrics 
% first modify them to have nan instead of zeros 
for i=1:length(hier_between_class)
    A=double(hier_within_class{i});
    A(A==0)=nan;
    hier_within_class{i}=A;
    
    B=double(hier_between_class{i});
    B(B==0)=nan;
    hier_between_class{i}=B;
    
end

Alphas=[];Gammas=[];
for i=1:length(hier_between_class)
    alpha=nanmean(reshape(res.data_covar.*(hier_within_class{i}),1,[]));
    gamma=nanmean(reshape(res.data_covar.*(hier_between_class{i}),1,[]));    
    Alphas(i)=alpha;
    Gammas(i)=gamma;
end
    Alphas=[mean(diag(res.data_covar)),Alphas];
    Gammas=[Alphas(2),Gammas];
    Alpha_cell{n}=Alphas;
    Gamma_cell{n}=Gammas;
    Cov_cell{n}=res.data_covar;
end

%% 
difference=cell2mat(Alpha_cell')-cell2mat(Gamma_cell');
figure;plot(beta_vals,difference)
set(gca,'xscale','log');

%% 
for n=1:length(Cov_cell)
cm=inferno(256);figure;imagesc(Cov_cell{n});caxis([0,1]);colormap(cm);
end 

data_cov=plot_tree_decomp(res.data)
