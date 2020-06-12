beta=0.4;
n_feat=2000;
n_entites=16;

% create a partition graph with 2 groups , one has 6 nodes and the other
% has 4 nodes 
% first level 
first_l=reshape(1:n_entites,2,[])';
first_lat=n_entites+(1:size(first_l,1))';
secnd_l=reshape(first_lat,2,[])';
second_lat=max(first_lat)+(1:size(secnd_l,1))';
third_l=reshape(second_lat,2,[])';
third_lat=max(second_lat)+(1:size(third_l,1))';
forth_l=reshape(third_lat,2,[])';
forth_lat=max(third_lat)+(1:size(forth_l,1))';

node_s=vertcat(first_l,secnd_l,third_l,forth_l);
node_t=vertcat(first_lat,second_lat,third_lat,forth_lat)*[1,1]

Gr=graph(node_s,node_t);
n_latent=Gr.numnodes-n_entites;
e = Gr.Edges;
plot(Gr);
adj_mat=full(adjacency(Gr));
figure;
imagesc(adj_mat);
%% 
F_mat=[]; 
S=sparse(exprnd(beta).*adj_mat);
sigma=5%exprnd(beta);
for n=1:n_feat
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
    F_mat=[F_mat;f];
end 
data=F_mat(:,1:n_entites)';
adj=adj_mat;
adjcluster=adj_mat((n_entites+1):end,(n_entites+1):end);

nobj=n_entites;
sigma=sigma;
G=inv(Delta_tilde);
structure='latenttree';
occind=repmat([1:size(first_l,1)],2,1);
occind=reshape(occind,1,[]);
save('/Users/eghbalhosseini/MyCodes/formdiscovery1.0_matlabR2014b/data/synthtree_eh.mat','data','adj','adjcluster','nobj','sigma','G','structure','occind');