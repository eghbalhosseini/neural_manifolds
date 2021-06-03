function create_figure_1_ICML_v2_local(file_to_pick)
save_path='~/MyData/neural_manifolds/data/';
plot_path='~/MyData/neural_manifolds/data/plots/';
%% construct file names to grab
file_pattern='synth_tree_nobj_64000_nclass_64_nhier_6_nfeat_936_beta*_norm_kemp_1_compressed.mat';
d_files=dir(strcat(save_path,filesep,file_pattern));
fprintf('file id : %d\n',file_to_pick)
fprintf('all files found : %d\n',length(d_files))

Alphas_list=[];
Gammas_list=[];
beta_vals=[];
for n=1:length(d_files)
    fprintf('%d\n',n)
    res=load(strcat(d_files(n).folder,filesep,d_files(n).name));
    if n==file_to_pick
        res_out=res;
    end
    Alphas_list=[Alphas_list;...
        res.ops_comp.Alphas];
    Gammas_list=[Gammas_list;...
        res.ops_comp.Gammas];
    beta_vals=[beta_vals;res.ops_comp.beta];
end

res=res_out.ops_comp;
res.data_covar=calc_cov(res.data);
difference=Alphas_list-Gammas_list;
%% figure setup

f=figure('visible','on');
u=f.Units;
f.Units='inches';
%f.Resize='off';
f.Position=[16.1944 2.9861 8 11];
xy_ratio=8/11;
rot=0;
pi_range=linspace(0+rot,2*pi+rot-pi/res.graph.class_graph.numnodes,res.graph.class_graph.numnodes);
x_starts=sin(pi_range);
y_starts=cos(pi_range);

% figure 1.A
ax=axes('position',[.001,.75,.25,.25]);
NodeCData=[ones(res.n_class,1)*[0,0,0];ones(res.graph.class_graph.numnodes-res.n_class,1)*[.4,.4,.4]];
g=plot(res.graph.class_graph,'layout','force','Iterations',50000,'UseGravity',true,'XStart',x_starts,'YStart',y_starts,'linewidth',2,'MarkerSize',3,'NodeColor',NodeCData,'NodeLabel',{});
%direction = [0 0 1];
%rotate(g,direction,25);
daspect([1,1,1]);
ax.Box='off';axis off;

%% figure 1.B
ax=axes('position',[.01,.55,.27,.27*xy_ratio]);
cm=inferno(512);
im=imagesc(res.data_covar);
caxis([0,.5]);
colormap(cm);
colorbar()
hold on;
ax.Box='off';axis off;

%% 1.C

% manually compute the graph class plots
within_class_ids=cat(2,[1:length(res.class_id)],res.hierarchical_class_ids);
between_class_ids=cat(2,res.hierarchical_class_ids,[1:length(res.class_id)]*0+1);
within_class_ids=cellfun(@(x) cast(x,'single'),within_class_ids,'uni',false);
between_class_ids=cellfun(@(x) cast(x,'single'),between_class_ids,'uni',false);


% 1st
n=1;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next

ax1=axes('position',[.32,.88,.15,.15*xy_ratio]);
mymap = [0 0 0; 1 .3 .3; .3 .3 1];
m=imagesc(ax1,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true)
m1=imagesc(ax1,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true)
ax1.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])
clear hier_within_class hier_between_class
% 2nd
n=2;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next
%
ax2=axes('position',[.475,.88,.15,.15*xy_ratio]);
m=imagesc(ax2,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true)
m1=imagesc(ax2,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true)
ax2.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])
clear hier_within_class hier_between_class
% 3rd
n=3;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next
%
ax3=axes('position',[.63,.88,.15,.15*xy_ratio]);

m=imagesc(ax3,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true);
m1=imagesc(ax3,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true);
ax3.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[]);
clear hier_within_class hier_between_class
% 4th
n=4;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next
%
ax1=axes('position',[.32,.765,.15,.15*xy_ratio]);
mymap = [0 0 0; 1 .3 .3; .3 .3 1];
m=imagesc(ax1,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true)
m1=imagesc(ax1,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true)
ax1.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])
clear hier_within_class hier_between_class
% 5th
n=5;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next

ax2=axes('position',[.475,.765,.15,.15*xy_ratio]);
m=imagesc(ax2,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true)
m1=imagesc(ax2,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true)
ax2.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])
clear hier_within_class hier_between_class
% 6th
n=6;
temp=within_class_ids{n}'*within_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class=(temp==temp1);
temp=within_class_ids{n+1}'*within_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
within_class_next=(temp==temp1);
temp=between_class_ids{n}'*between_class_ids{n};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class=(temp==temp1);
temp=between_class_ids{n+1}'*between_class_ids{n+1};
temp1=repmat(diag(temp),1,length(res.class_id));
between_class_next=(temp==temp1);
hier_within_class=xor(within_class,within_class_next);
hier_between_class=xor(between_class,between_class_next);
clear temp temp1 within_class within_class_next between_class between_class_next

ax3=axes('position',[.63,.765,.15,.15*xy_ratio]);

m=imagesc(ax3,hier_within_class);
hold on
set(m,'AlphaData',hier_within_class==true)
m1=imagesc(ax3,2*hier_between_class);
set(m1,'AlphaData',hier_between_class==true)
ax3.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[]);
clear hier_within_class hier_between_class
%%
% first sort the betas 
[beta_vals_sort,sort_idx]=sort(beta_vals);
difference_sort=difference(sort_idx,:);
% plot the
line_cols=viridis(size(difference,2)+1);
ax6=axes('position',[.36,.55,.42,0.2]);
hold on;
beta_res=arrayfun(@(x) plot(beta_vals_sort,difference_sort(:,x),'color',line_cols(x,:),...
    'displayname',sprintf('hierarchy %d',x),'markersize',10),1:size(difference_sort,2));

arrayfun(@(x) set(beta_res(x),'LineWidth',2),1:length(beta_res))
set(gca,'yscale','linear');
set(gca,'xscale','log');

ax6.XLim=[beta_vals_sort(1),beta_vals_sort(end)];
ax6.Box='off';
ax6.XAxis.FontSize=12;
ax6.XAxis.LineWidth=1;
ax6.YAxis.FontSize=12;
ax6.YAxis.LineWidth=1;
legend('position',[.78,.64,.1,.1])
ax6.Legend.FontSize=12;
ylabel('$\bf{\alpha}-\bf{\gamma}$','fontsize',14,'interpreter','latex');
xlabel('$\bf{\beta}$','fontsize',14,'interpreter','latex','rotation',0);
% plot the selected beta value
plot(res.beta*[1,1],ax6.YLim,'color',[.5,.5,.5],'linewidth',1,'displayname',sprintf('beta=%f',res.beta));
axis tight
%%
file_id=find(beta_vals_sort==res.beta);
temp1=res.data_id(1:regexp(res.data_id,'_beta_'));
plt_name_str=strcat(plot_path,filesep,...
    sprintf("%sbeta_vals_%d_%d_sigma_%d_id_%d_beta_plot_%d.pdf",temp1,beta_vals_sort(1),beta_vals_sort(end),res.sigma,file_id,res.beta));
print(f,'-painters', '-dpdf', plt_name_str);


fprintf("saved file at %s",plt_name_str);

end

