n_class=80;
F=true;
rec=n_class;
br=[];
while F
    rec_div=divisors(rec);
    if sum(ismember(rec_div,2))
        rec=rec/2;
        br=[br,2];
    else
        br=[br,rec];
        F=false;
    end 
    
end 
br=fliplr(br);

node_s=[];
node_t=[];
groups=transpose(1:n_class);
for k=1:length(br)-1
    node_s=[node_s;groups]
    groups=reshape(groups,[],length(groups)/br(k));
    links=[1:size(groups,2)]+max(node_s);
    groups=links';
    links=repmat(links,br(k),1);
    node_t=[node_t;links(:)];
end 
node_s=[node_s;groups(1)];
node_t=[node_t;groups(2)];

G=graph(node_s,node_t);
plot(G,'layout','force3');