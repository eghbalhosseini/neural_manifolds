function ops_out=neural_manifold_create_partition_dataset_cholesky_method(varargin)
% create a partition dataset based on gaussian prior 
% usage neural_manifold_create_partition_dataset_cholesky_method()

% Eghbal Hosseini, 16/06/2020
% parse inputs 
p=inputParser();
addParameter(p, 'n_class', 10);
addParameter(p, 'exm_per_class', 100);
addParameter(p, 'n_feat', 28*28);
addParameter(p, 'beta', 0.4);
addParameter(p, 'sigma', 0.5);
addParameter(p, 'save_path', '~/');
parse(p, varargin{:});
ops = p.Results;
ops_out=ops;
% construct a partition graph  
n_ent=ops.n_class.*ops.exm_per_class;
n_latent=ops.n_class;
node_s=1:n_ent;
node_t=ones(ops.exm_per_class,1)*[n_ent+(1:ops.n_class)];
node_t=reshape(node_t,1,[]);
Gr = graph(node_s,node_t);
adj=(adjacency(Gr));
class_id=ones(ops.exm_per_class,1)*[(1:ops.n_class)];
class_id=reshape(class_id,1,[]);
structure='flat';
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
save(strcat(ops.save_path,'neural_manifold_synthpartition.mat'),'ops_out');
fprintf('saved data in %s \n',strcat(ops.save_path,'neural_manifold_synthpartition.mat'));
end