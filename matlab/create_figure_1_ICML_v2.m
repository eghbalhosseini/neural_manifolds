clear all 
close all 
% data creation
n_class=64;
exm_per_class=5;
n_feat=50;
beta=.05;
res=create_synth_data_cholesky_method('structure','tree','n_class',n_class,'exm_per_class',exm_per_class,'n_feat',n_feat,'beta',beta,'sigma',5,'norm',true,'save',false);
res.data_covar=cov(res.data');
res=compute_class_distance(res);

%% figure setup
f=figure;
u=f.Units;
f.Units='inches';
%f.Resize='off';
f.Position=[16.1944 2.9861 8 11];
xy_ratio=8/11;

pi_range=linspace(0,2*pi-pi/res.graph.class_graph.numnodes,res.graph.class_graph.numnodes);
x_starts=sin(pi_range);
y_starts=cos(pi_range);

% figure 1.A 
ax=axes('position',[.001,.75,.25,.25]);
NodeCData=[ones(res.n_class,1)*[0,0,0];ones(res.graph.class_graph.numnodes-res.n_class,1)*[.4,.4,.4]];
g=plot(res.graph.class_graph,'layout','force','Iterations',50000,'UseGravity',true,'XStart',x_starts,'YStart',y_starts,'linewidth',2,'MarkerSize',3,'NodeColor',NodeCData,'NodeLabel',{});
direction = [0 0 1];
rotate(g,direction,25);
daspect([1,1,1]);
ax.Box='off';axis off;

%% figure 1.B
ax=axes('position',[.01,.55,.27,.27*xy_ratio]);
cm=inferno(256);
im=imagesc(res.data_covar);
caxis([0,1]);
colormap(cm);
hold on;
ax.Box='off';axis off;
%% 1.C
% 1st 
ax1=axes('position',[.32,.88,.15,.15*xy_ratio]);
mymap = [0 0 0; 1 .3 .3; .3 .3 1];
A=double(res.hier_within_class{1});
A(A==0)=nan;
B=double(res.hier_between_class{1});
B(B==0)=nan;
m=imagesc(ax1,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax1,2*B);
set(m1,'AlphaData',~isnan(B))
ax1.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])

% 2nd 
ax2=axes('position',[.475,.88,.15,.15*xy_ratio]);
A=double(res.hier_within_class{2});
A(A==0)=nan;
B=double(res.hier_between_class{2});
B(B==0)=nan;
m=imagesc(ax2,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax2,2*B);
set(m1,'AlphaData',~isnan(B))
ax2.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])

% 3rd
ax3=axes('position',[.63,.88,.15,.15*xy_ratio]);
A=double(res.hier_within_class{3});
A(A==0)=nan;
B=double(res.hier_between_class{3});
B(B==0)=nan;
m=imagesc(ax3,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax3,2*B);
set(m1,'AlphaData',~isnan(B))
ax3.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[]);

% 4th 
ax1=axes('position',[.32,.765,.15,.15*xy_ratio]);
mymap = [0 0 0; 1 .3 .3; .3 .3 1];
A=double(res.hier_within_class{4});
A(A==0)=nan;
B=double(res.hier_between_class{4});
B(B==0)=nan;
m=imagesc(ax1,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax1,2*B);
set(m1,'AlphaData',~isnan(B))
ax1.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])

% 5th 
ax2=axes('position',[.475,.765,.15,.15*xy_ratio]);
A=double(res.hier_within_class{5});
A(A==0)=nan;
B=double(res.hier_between_class{5});
B(B==0)=nan;
m=imagesc(ax2,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax2,2*B);
set(m1,'AlphaData',~isnan(B))
ax2.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[])

% 6th
ax3=axes('position',[.63,.765,.15,.15*xy_ratio]);
A=double(res.hier_within_class{6});
A(A==0)=nan;
B=double(res.hier_between_class{6});
B(B==0)=nan;
m=imagesc(ax3,A);
hold on 
set(m,'AlphaData',~isnan(A))
m1=imagesc(ax3,2*B);
set(m1,'AlphaData',~isnan(B))
ax3.Colormap=mymap;
caxis([0 3]);
set(gca,'XTick',[],'YTick',[]);


%A_fixer=tril(ones(size(A)));
%A_fixer(A_fixer==1)=nan;
%A=A+A_fixer;
%A1=double(res.hier_within_class{2});
%A1(A1==0)=nan;
%A1=A1+A_fixer;

% s=pcolor(ax1,1*A)
% s.EdgeAlpha=0;
% s.FaceAlpha=1;
% hold on 
% s=pcolor(ax1,2*A1);
% s.EdgeAlpha=0;
% s.FaceAlpha=1;

% ax1.Box='off';axis off;
% ax1.YDir='reverse';
% set(ax1,'color','none');
% mymap = [1 1 0; 0 1 1; 0 0 1; 1 1 0; 0 0 0];
% ax1.Colormap=mymap;
% caxis([1 5])

%% 
% figure 
%  a = rand(5);
%  a(3,3) = NaN;
%  b = imagesc(a);
%  set(b,'AlphaData',~isnan(a))
% do a placeholder for plot 


% plot alphas and gammas for different betas, 
beta_vals=logspace(-5,1.5,50);
Alpha_cell={};
Gamma_cell={};
Cov_cell={};
for n=1:length(beta_vals)
res=create_synth_data_cholesky_method('structure','tree','n_class',16,'exm_per_class',5,'n_feat',100,'beta',beta_vals(n),'sigma',5,'norm',true,'save',false);
within_class_ids=cat(2,[1:length(res.class_id)],res.hierarchical_class_ids);
between_class_ids=cat(2,res.hierarchical_class_ids,[1:length(res.class_id)]*0+1);
res.data_covar=cov(res.data');

within_cell={};
between_cell={};
for i=1:length(within_class_ids)
    temp=within_class_ids{i}'*within_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    within_class=double(arrayfun(@(x,y) isequal(x,y),temp,temp1));
    within_cell{i}=within_class;
    % do the same to get between class 
    temp=between_class_ids{i}'*between_class_ids{i};
    temp1=repmat(diag(temp),1,length(res.class_id));
    between_class=double(arrayfun(@(x,y) isequal(x,y),temp,temp1));
    between_cell{i}=between_class;
    
    
end 
hier_within_class={};
hier_between_class={};

for i=1:length(within_class_ids)-1
    A=(arrayfun(@(x,y) xor(x,y),within_cell{i}, within_cell{i+1}));
    hier_within_class{i}=A;
    
    B=(arrayfun(@(x,y) xor(x,y),between_cell{i}, between_cell{i+1}));
    hier_between_class{i}=B;
end 

% compute the metrics 
% first modify them to have nan instead of zeros 
for i=1:length(hier_between_class)
    A=double(hier_within_class{i});
    A(A==0)=nan;
    hier_within_class{i}=A;
    
    B=double(hier_between_class{i});
    B(B==0)=nan;
    hier_between_class{i}=B;
    
end

Alphas=[];Gammas=[];
for i=1:length(hier_between_class)
    alpha=nanmean(reshape(res.data_covar.*(hier_within_class{i}),1,[]));
    gamma=nanmean(reshape(res.data_covar.*(hier_between_class{i}),1,[]));    
    Alphas(i)=alpha;
    Gammas(i)=gamma;
end
    Alphas=[mean(diag(res.data_covar)),Alphas];
    Gammas=[Alphas(2),Gammas];
    Alpha_cell{n}=Alphas;
    Gamma_cell{n}=Gammas;
    Cov_cell{n}=res.data_covar;
end

%
difference=cell2mat(Alpha_cell')-cell2mat(Gamma_cell');
%% 
% 6th
line_cols=viridis(size(difference,2)+1);
ax6=axes('position',[.36,.55,.42,0.2]);
hold on;
beta_res=arrayfun(@(x) plot(beta_vals,difference(:,x),'color',line_cols(x,:),...
    'displayname',sprintf('hierarchy %d',x)),1:size(difference,2));

arrayfun(@(x) set(beta_res(x),'LineWidth',2),1:length(beta_res))
set(gca,'yscale','linear');
set(gca,'xscale','log');

ax6.XLim=[beta_vals(1),beta_vals(end)];
ax6.Box='off';
ax6.XAxis.FontSize=12;
ax6.XAxis.LineWidth=1;
ax6.YAxis.FontSize=12;
ax6.YAxis.LineWidth=1;
legend('position',[.42,.64,.1,.1])
ax6.Legend.FontSize=12;
ylabel('$\bf{\alpha}-\bf{\gamma}$','fontsize',14,'interpreter','latex')
xlabel('$\bf{\beta}$','fontsize',14,'interpreter','latex','rotation',0)

%% 
cdr=cd;
print(f,'-painters', '-dpdf', strcat(cd,'/',res.data_id(1:end-4),'_','fig_1_v2','.pdf')); 


