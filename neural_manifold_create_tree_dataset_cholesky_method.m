function ops_out=neural_manifold_create_tree_dataset_cholesky_method(varargin)
% create a partition dataset based on gaussian prior 
% usage neural_manifold_create_partition_dataset_cholesky_method()

% Eghbal Hosseini, 16/06/2020
% for now its fixed with a certain number of classes. 
% parse inputs  
p=inputParser();
addParameter(p, 'n_class', 50);
addParameter(p, 'exm_per_class', 1000);
addParameter(p, 'class_depth', 3);
addParameter(p, 'n_feat', 28*28);
addParameter(p, 'beta', 0.4);
addParameter(p, 'sigma', 0.5);
addParameter(p, 'save_path', '~/');
parse(p, varargin{:});
ops = p.Results;
ops_out=ops;
% construct a partition graph  
first_level_class=10;
first_example_per_class=5;

second_level_class=2;
second_example_per_class=5;


n_ent=ops.n_class.*ops.exm_per_class;
example_per_class=ops.exm_per_class;
num_classes=ops.n_class;
% first level 
first_s=1:n_ent;
first_t=ones(example_per_class,1)*[n_ent+(1:num_classes)];
first_t=reshape(first_t,1,[]);
first_lat=unique(first_t);
second_s=first_lat;
second_t=ones(first_example_per_class,1)*[max(first_lat)+(1:first_level_class)];
second_t=reshape(second_t,1,[]);
second_lat=unique(second_t);
third_s=second_lat;
third_t=ones(second_example_per_class,1)*[max(second_lat)+(1:second_level_class)];
third_t=reshape(third_t,1,[]);
third_lat=unique(third_t)
node_s=horzcat(first_s,second_s,third_s);
node_t=horzcat(first_t,second_t,third_t);
Gr=graph([first_s,second_s,third_s],[first_t,second_t,third_t]);
Gr=addedge(Gr,third_lat(1),third_lat(2));
adj=(adjacency(Gr));
n_latent=Gr.numnodes-n_ent;
% create class ids 
class_id=ones(ops.exm_per_class,1)*[(1:ops.n_class)];
class_id=reshape(class_id,1,[]);
structure='latenttree';
% create a feature dataset based on graph
F_mat=nan*ones(n_ent+n_latent,ops.n_feat); 
Delta_tilde=sparse(zeros(size(adj,1),size(adj,2)));
beta=ops.beta;
sigma=ops.sigma;
n_feat=ops.n_feat;
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
    F_mat(:,n) = L_Lambda'\z;
    fprintf('feature: %d\n',n);
end
% save the results 
ops_out=ops;
ops_out.data=F_mat(1:n_ent,:);
ops_out.data_latent=F_mat((n_ent+1):end,:);
ops_out.Adjacency=adj;
ops_out.struct=structure;
ops_out.n_latent=n_latent;
ops_out.class_id=class_id;
save(strcat(ops.save_path,sprintf('synthtree_nobj_%d_nclass_%d_nfeat_%d.mat',n_ent,ops.n_class,n_feat)),'ops_out');
fprintf('saved data in %s \n',strcat(ops.save_path,sprintf('synthpartition_nobj_%d_nclass_%d_nfeat_%d.mat',n_ent,ops.n_class,n_feat)));
end