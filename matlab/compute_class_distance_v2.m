%% 
function res_out=compute_class_distance_v2(res)

res_out=res;
% append the 0th level 
within_class_ids=cat(2,[1:length(res.class_id)],res.hierarchical_class_ids);
between_class_ids=cat(2,res.hierarchical_class_ids,[1:length(res.class_id)]*0+1);
% for efficientcy convert within class and between class to uint32
%within_class_ids=cellfun(@(x) cast(x,'uint32'),within_class_ids,'uni',false);
%between_class_ids=cellfun(@(x) cast(x,'uint32'),between_class_ids,'uni',false);
%

Alphas=[];
Gammas=[];
for i=1:length(within_class_ids)-1
    fprintf('constructing masks for hierarchy % d\n',i)
    temp=within_class_ids{i}'*within_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    within_class=(temp==temp1);
    % 
    temp=within_class_ids{i+1}'*within_class_ids{i+1};
    temp1=repmat(diag(temp),1,length(res.class_id));
    within_class_next=(temp==temp1);
    
    
    % do the same to get between class 
    temp=between_class_ids{i}'*between_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    between_class=(temp==temp1);
    % 
    temp=between_class_ids{i+1}'*between_class_ids{i+1};
    temp1=repmat(diag(temp),1,length(res.class_id));
    between_class_next=(temp==temp1);
    % 
    hier_within_class=xor(within_class,within_class_next);
    hier_between_class=xor(between_class,between_class_next);
    % clear some space
    clear temp temp1 within_class within_class_next between_class between_class_next
    % 
    fprintf('computing alpha and gamma for hierarchy %d\n',i)
    alpha=nanmean(res.data_covar(hier_within_class));
    gamma=nanmean(res.data_covar(hier_between_class));   
    Alphas(i)=alpha;
    Gammas(i)=gamma; 
end 

res_out.Alphas=Alphas;
res_out.Gammas=Gammas;
end 

