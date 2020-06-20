function ops_out=neural_manifold_create_synth_data_cholesky_method(varargin)
% create synthetic structured dataset based on gaussian prior 
% ref: Kemp, Charles, and Joshua B. Tenenbaum. 2008. ?The Discovery of Structural Form.? PNAS.
% note for tree case it creates a fractal patten tree. 
% usage neural_manifold_create_synth_dataset_cholesky_method()

% Eghbal Hosseini, 20/06/2020
% changelog: 
% TODO: fix the routine for making tree
% parse inputs  
p=inputParser();
addParameter(p, 'n_class', 50);
addParameter(p, 'exm_per_class', 1000);
addParameter(p, 'class_depth', 3);
addParameter(p, 'n_feat', 28*28);
addParameter(p, 'beta', 0.01);
addParameter(p, 'structure', 'partition'); % options : partition , tree
addParameter(p, 'sigma', 1.5);
addParameter(p,'norm',true);
addParameter(p, 'save_path', '~/');
parse(p, varargin{:});
ops = p.Results;
% 
beta=ops.beta;
sigma=ops.sigma;
n_feat=ops.n_feat;
n_ent=floor(ops.n_class.*ops.exm_per_class);
ex_pr_cl=ops.exm_per_class;
n_cl=ops.n_class;
is_norm=ops.norm;
% construct a partition graph  
%   - Hr is the graph for how classes are connected, in case of partition is it
%   is nan
%   - Gr is the complete graph for the entities and classes. 
[Gr,Hr]=create_graph_for_structure(ops);
% first level 
adj=(adjacency(Gr));
n_latent=Gr.numnodes-n_ent;
% create class ids 
class_id=ones(ex_pr_cl,1)*(1:n_cl);
class_id=reshape(class_id,1,[]);
% create a feature dataset based on graph
F_mat=nan*ones(n_ent+n_latent,n_feat); 
V=sparse(diag([(1/sigma^2)*ones(1,n_ent),zeros(1,n_latent)]));
parfor n=1:n_feat
    S=sparse(exprnd(beta).*adj);
    W=(spfun(@(x) 1./x,S));
    E=diag(sum(W,2));
    % graph laplacian
    Delta=E-W;
    % proper prior
    Delta_tilde=Delta+V;
    % univariate random
    z = randn(n_ent+n_latent,1); 
    L_Lambda = chol(Delta_tilde,'lower'); 
    dat_feat=L_Lambda'\z;
    if is_norm
        dat_feat = (dat_feat - min(dat_feat)) / ( max(dat_feat) - min(dat_feat));
    end 
    F_mat(:,n) = dat_feat;
    fprintf('feature: %d\n',n);
end
% save the results 
ops_out=ops;
ops_out.data=F_mat(1:n_ent,:);
ops_out.data_latent=F_mat((n_ent+1):end,:);
ops_out.Adjacency=adj;
ops_out.n_latent=n_latent;
ops_out.class_id=class_id;
ops_out.entity_graph=Gr;
ops_out.class_graph=Hr;
data_loc=strcat(ops.save_path,sprintf('synth%s_nobj_%d_nclass_%d_nfeat_%d_norm_%d.mat',ops.structure,n_ent,n_cl,n_feat,is_norm));
save(data_loc,'ops_out','-v7.3');
fprintf('saved data in %s \n',data_loc);
end
% funtion for making the graph 
function [Gr,Hr]=create_graph_for_structure(ops)
    gr_struct=ops.structure;
    n_ent=floor(ops.n_class.*ops.exm_per_class);
    ex_pr_cl=ops.exm_per_class;
    n_cl=ops.n_class;
    switch gr_struct
        case 'partition'
            % construct a partition graph  
            node_s=1:n_ent;
            node_t=ones(ex_pr_cl,1)*n_ent+(1:n_cl);
            node_t=reshape(node_t,1,[]);
            Gr = graph(node_s,node_t);
            Hr = nan; 
        case 'tree'
            if isEven(ops.n_class)
                
                % create a fractal tree: 
                n_temp=ops.n_class;
                i=2;
                divs=[];
                while n_temp>1
                    while ~mod(n_temp,i)
                        n_temp=n_temp/i;
                        divs=[divs,i];
                    end
                    i=i+1;
                end
                % create node list 
                divs=fliplr(divs(2:end));
                n_temp=n_cl;
                new_nodes={n_temp};
                for k=divs
                    n_temp=n_temp/k;
                    new_nodes=[new_nodes,n_temp];
                end
                divs_temp=[divs,1];
                n_t_plus=cumsum(cell2mat(new_nodes));
                n_s_plus=[0,n_t_plus(1:end-1)];
                n_s_cell=arrayfun(@(x) (1:new_nodes{x}),1:size(new_nodes,2),'uni',false);
                n_t_cell=arrayfun(@(x) reshape(n_s_cell{x},divs_temp(x),[]),1:size(new_nodes,2),'uni',false);
                
                n_t_cell_new=arrayfun(@(x) repmat(1:size(n_t_cell{x},2),size(n_t_cell{x},1),1)+n_t_plus(x),1:size(n_t_cell,2),'uni',false);
                n_s_cell_new=arrayfun(@(x) n_s_cell{x}+n_s_plus(x),1:size(n_s_cell,2),'uni',false);
                n_t=cell2mat(cellfun(@(x) transpose(reshape(x,[],1)),n_t_cell_new,'uni',false));
                % connect 2 sets
                n_t(end)=n_t(end-1);
                n_s=cell2mat(n_s_cell_new);
                % tree structure
                Hr=graph(n_s,n_t);
                % add the entities to the graph
                first_s=1:n_ent;
                first_t=ones(ex_pr_cl,1)*n_ent+(1:n_cl);
                first_t=reshape(first_t,1,[]);
                n_s=horzcat(first_s,n_ent+n_s);
                n_t=horzcat(first_t,n_ent+n_t);
                Gr=graph(n_s,n_t);
            else
                error('tree is only valid for even number of classes')
            end 
        otherwise 
            error('unknown structure type')
    end 

end 