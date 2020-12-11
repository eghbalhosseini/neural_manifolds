function ops_out=create_synth_data_cholesky_method_draft(varargin)
% create synthetic structured dataset based on gaussian prior 
% ref: Kemp, Charles, and Joshua B. Tenenbaum. 2008. ?The Discovery of Structural Form.? PNAS.
% note for tree case it creates a fractal patten tree. 
% usage neural_manifold_create_synth_dataset_cholesky_method()

% Eghbal Hosseini, 20/06/2020
% changelog: GT + EH, nov25th etc.
% TODO: fix the routine for making tree
% parse inputs  
p=inputParser();
addParameter(p, 'n_class', 10);
addParameter(p, 'exm_per_class', 500);
addParameter(p, 'n_feat', 500);
addParameter(p, 'beta', 0.4 );
addParameter(p, 'structure', 'partition'); % options : partition , tree
addParameter(p, 'sigma', 5);
addParameter(p,'norm',false);
addParameter(p, 'save_path', '~/');
addParameter(p, 'save', false);
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
[gr_output]=create_graph_for_structure(ops);

% first level 
adj=(adjacency(gr_output.full_graph));
n_latent=gr_output.full_graph.numnodes-n_ent;
n_hier=size(gr_output.class_ids,2);

% create a feature dataset based on graph
F_mat=nan*ones(n_ent+n_latent, n_feat); % initialize
F_mat_norm=nan*ones(n_ent+n_latent, n_feat); % initialize
%
%index=find(adj);
%S=adj;
%Beta_val=exprnd(beta,length(index),1);
%for iter=1:length(index)
    %S(index(iter))=Beta_val(iter);
%    S(index(iter))=beta;
%end
%S=triu(S); % draw from exp. distribution using beta parameter
%S=S+S';
%Adj=(spfun(@(x) 1./x,S));
%Degree=diag(sum(Adj,2));
% graph laplacian : needs to be positive definite
%Laplacian=Degree-Adj;
% proper prior
for n=1:n_feat
    Beta_val=exprnd(beta);
    Sigma_val=exprnd(sigma);
    S=triu(sparse(Beta_val.*adj));
    S=S+S';
    Adj=(spfun(@(x) 1./x,S));
    Degree=diag(sum(Adj,2));
    Laplacian=Degree-Adj;
    %V = spdiags([(1/(Sigma_val^2))*ones(1,n_ent),zeros(1,n_latent)]',0,n_ent+n_latent,n_ent+n_latent); % first part is 1/sigma^2 I and then L, the graph structure
    V = spdiags([(1/(sigma^2))*ones(1,n_ent),zeros(1,n_latent)]',0,n_ent+n_latent,n_ent+n_latent); % first part is 1/sigma^2 I and then L, the graph structure
    %S=adj;
    %Beta_val=exprnd(beta,length(index),1);
    Laplacian_tilde=Laplacian+V;
    %for iter=1:length(index)
    %    S(index(iter))=Beta_val(iter);
    %end 
    % adjacency matrix needs to symmetric 
    % proper prior
    %L_lambda_inv=pdinv(Laplacian_tilde);
    % univariate random
    z = randn(n_ent+n_latent,1); 
    L_Lambda = chol(Laplacian_tilde,'lower'); 
    dat_feat=L_Lambda'\z; % solves system of equation for X= mu + AZ where mu=0 
    %dat_feat=L_Lambda*z;
    %dat_feat = mvnrnd(0*ones(1,n_ent+n_latent),L_lambda_inv) ;
    dat_feat_norm = (dat_feat - min(dat_feat)) / ( max(dat_feat) - min(dat_feat));
    if is_norm
        dat_feat = (dat_feat - min(dat_feat)) / ( max(dat_feat) - min(dat_feat));
    end 
    
    F_mat(:,n) = dat_feat;
    F_mat_norm(:,n) = dat_feat_norm;
    fprintf('feature: %d\n',n);
end
% save the results 
ops_out=ops;
ops_out.data=F_mat(1:n_ent,:);
ops_out.data_scaled_kemp=rescale_data(F_mat(1:n_ent,:));
ops_out.data_scaled_G_E=F_mat_norm(1:n_ent,:);
ops_out.data_latent=F_mat((n_ent+1):end,:);
ops_out.Adjacency=adj;
ops_out.weight=S;
ops_out.beta_vals=Beta_val;
ops_out.n_latent=n_latent;
ops_out.hierarchical_class_ids=gr_output.class_ids;
ops_out.class_id=gr_output.class_ids{1};
ops_out.graph=gr_output;
ops_out.n_hier=n_hier;
data_loc=strcat(ops.save_path,sprintf('synth_%s_nobj_%d_nclass_%d_nhier_%d_nfeat_%d_beta_%1.4f_sigma_%1.4f_norm_%d.mat',ops.structure,n_ent,n_cl,n_hier,n_feat,beta,sigma,is_norm));
if ops.save
save(data_loc,'ops_out','-v7.3');
fprintf('saved data in %s \n',data_loc);
end 

end
% funtion for making the graph 
function [gr_output]=create_graph_for_structure(ops)
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
            class_id=ones(ex_pr_cl,1)*(1:n_cl);
            class_ids={reshape(class_id,1,[])};
        case 'tree'
            if ~mod(ops.n_class,2)
                
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
                % create class_ids
                class_ids=cellfun(@(x) reshape(reshape(ones(1,n_ent),[],length(x))*diag(x),[],1),n_s_cell,'uni',false);
                class_ids=cellfun(@transpose,class_ids,'uni',false);
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
    gr_output=struct;
    gr_output.full_graph=Gr;
    gr_output.class_graph=Hr;
    gr_output.class_ids=class_ids;
end 