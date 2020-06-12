beta=0.4;
n_feat=28*28;
num_classes=50;
example_per_class=400;
n_entites=num_classes.*example_per_class;
n_latent=num_classes;
node_s=1:n_entites;
node_t=ones(example_per_class,1)*[n_entites+(1:num_classes)];
node_t=reshape(node_t,1,[]);
% create a partition graph with 2 groups , one has 6 nodes and the other
% has 4 nodes 
Gr = graph(node_s,node_t);
e = Gr.Edges;
plot(Gr);
adj_mat=full(adjacency(Gr));
figure;
imagesc(adj_mat)
%% 
F_mat=nan*ones(n_feat,n_entites+n_latent); 
S=sparse(exprnd(beta).*adj_mat);
Delta_tilde=0;
sigma=5;%exprnd(beta);
parfor n=1:n_feat
    S=sparse(exprnd(beta).*adj_mat);
    V=diag([(1/sigma^2)*ones(1,n_entites),zeros(1,n_latent)]);
    W=full(spfun(@(x) 1./x,S));
    E=diag(sum(W,2));
    % graph laplacian
    Delta=E-W;
    % proper prior
    Delta_tilde=Delta+V;
    %imagesc(inv(Delta_tilde))
    f=mvnrnd(zeros(1,n_entites+n_latent),inv(Delta_tilde));
    F_mat(n,:)=[f];
    fprintf('%d\n',n)
end 
F_mat=F_mat';
data=F_mat(1:n_entites,:);
adj=adj_mat;
class_id=ones(example_per_class,1)*[(1:num_classes)];
class_id=reshape(class_id,1,[]);
adjcluster=zeros(num_classes,num_classes);
nobj=n_entites;
sigma=sigma;
G=inv(Delta_tilde);
structure='flat';
occind=node_t-n_entites;
%save('/Users/eghbalhosseini/MyCodes/formdiscovery1.0_matlabR2014b/data/synthpartition_eh.mat','data','adj','adjcluster','nobj','sigma','G','structure','occind');
save('/Users/eghbalhosseini/MyData/neural_manifolds/data/synthpartition_eh.mat','data','adj','adjcluster','nobj','sigma','G','structure','occind','F_mat','class_id');