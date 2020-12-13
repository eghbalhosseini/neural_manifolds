%% 
close all 
res=create_synth_data_cholesky_method('structure','tree','n_class',16,'exm_per_class',1000,'n_feat',936,'beta',.4,'sigma',1,'norm',true,'save',false);
% append the 0th level 
within_class_ids=cat(2,[1:length(res.class_id)],res.hierarchical_class_ids);
between_class_ids=cat(2,res.hierarchical_class_ids,[1:length(res.class_id)]*0+1);
res.data_covar=cov(res.data');
%
within_cell={};
between_cell={};
for i=1:length(within_class_ids)
    temp=within_class_ids{i}'*within_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    within_class=(arrayfun(@(x,y) isequal(x,y),temp,temp1));
    within_cell{i}=within_class;
    % do the same to get between class 
    temp=between_class_ids{i}'*between_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    between_class=(arrayfun(@(x,y) isequal(x,y),temp,temp1));
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
hier_within_class_logic=hier_within_class;
hier_between_class_logic=hier_between_class;
% 
f=figure;
set(f,'position',[32 775 1676 243]);
ax=subplot(1,length(within_class_ids),1);
cm=inferno(256);imagesc(res.data_covar);axis square
caxis([0,1]);colormap(cm);
title(sprintf('num of hierarchies = %d',res.n_hier))
for i=1:length(hier_between_class)
    ax=subplot(1,length(within_class_ids),i+1);
    
    imagesc(hier_within_class{i}-hier_between_class{i});
    colormap(ax,'gray');
    axis square
    title('within = white ; between = black, gray = excluded')
end


% compute the metrics 
% first modify them to have nan instead of zeros 
for i=1:length(hier_between_class)
    A=double(hier_within_class{i});
    A(A==0)=nan;
    hier_within_class{i}=sparse(A);
    
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

Alphas./Gammas

Alphas-Gammas

mean(diag(res.data_covar))

cm=inferno(256);figure;imagesc(res.data_covar);caxis([0,1]);colormap(cm);

% do it based on logical indexing 


Alphas_logic=[];Gammas_logic=[];
for i=1:length(hier_between_class)
    alpha=nanmean(res.data_covar(hier_within_class_logic{i}));
    gamma=nanmean(res.data_covar(hier_between_class_logic{i}));    
    Alphas_logic(i)=alpha;
    Gammas_logic(i)=gamma;
end


