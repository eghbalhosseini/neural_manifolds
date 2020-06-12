beta=0.4;
n_feat=1000;
num_classes=50;

example_per_class=100;

first_level_class=10;
first_example_per_class=5;

second_level_class=2;
second_example_per_class=5;


n_entites=num_classes.*example_per_class;

% has 4 nodes 
% first level 

first_s=1:n_entites;
first_t=ones(example_per_class,1)*[n_entites+(1:num_classes)];
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

Gr=addedge(Gr,third_lat(1),third_lat(2))
n_latent=Gr.numnodes-n_entites;
e = Gr.Edges;

figure
%subplot(1,2,1)
%H1 = subgraph(Gr,1:max(second_lat));
%plot(H1,'Layout','force');

H = subgraph(Gr,(n_entites+1):max(third_lat));

%subplot(1,2,2)
plot(H,'Layout','force');
set(gca,'fontsize',20)
adj_mat=full(adjacency(Gr));
figure;
imagesc(adj_mat);
%% 
F_mat=nan*ones(n_feat,n_entites+n_latent);
Delta_tilde=nan;
S=sparse(exprnd(beta).*adj_mat);
sigma=5%exprnd(beta);
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
class_id=ones(example_per_class,1)*[(1:num_classes)];
class_id=reshape(class_id,1,[]);


adj=adj_mat;
adjcluster=adj_mat((n_entites+1):end,(n_entites+1):end);

nobj=n_entites;
sigma=sigma;
G=inv(Delta_tilde);
structure='latenttree';

save('/Users/eghbalhosseini/MyData/neural_manifolds/data/synthtree_eh.mat','data','adj','adjcluster','nobj','sigma','G','structure','F_mat','class_id');