%% Figure parameters
set(0,'defaultfigurecolor',[1 1 1])

%%
% do one to get the within and between identifier 
ops=create_synth_data_cholesky_method('n_class',20,'n_feat',100,'exm_per_class',10,'save_path',join([pwd,'\synth_datasets']))
% n_class (20)*exm_per_class (10) = 200 data points, 100 features each.

figure;imagesc(ops.data)
c=cov(transpose(ops.data));
figure;imagesc(c)

temp=ops.class_id'*ops.class_id;
figure;imagesc(temp)

temp1=repmat(diag(temp),1,length(ops.class_id));
figure;imagesc(temp1)

within_class=double(arrayfun(@(x,y) isequal(x,y),temp,temp1));
between_class=double(~within_class);
within_class(within_class==0)=nan;
within_class=1+within_class.*(diag(nan*ones(length(within_class),1)));
between_class(between_class==0)=nan;

figure;imagesc(within_class)
figure;imagesc(between_class)

%% Function for visualizing various aspects of the data
plot_decomp(ops.data)

%% 
% beta_range=linspace(1e-10,.05,10);
% sigma_range=linspace(1e-5,2.5,10);

beta_range=linspace(1e-10,.05, 4);
sigma_range=linspace(1e-5,2.5, 4);

% beta_range=linspace(1e-10,5,10); % does not provide more info 
% sigma_range=linspace(1e-10,5,10); % does not provide more info

[Beta,Sigma]=meshgrid(beta_range,sigma_range);

c_ratio=nan*Beta;
BTW_c=c_ratio;
WTH_c=c_ratio;

% Variance
BTW_var=c_ratio;
WTH_var=c_ratio;

% pdist
BTW_pdist=c_ratio;
WTH_pdist=c_ratio;


for i=1:(length(Beta(:)))
    ops=create_synth_data_cholesky_method('n_class',20,'n_feat',100,'exm_per_class',10,'beta',Beta(i),'sigma',Sigma(i));
    c=cov(transpose(ops.data));
    
%     figure;imagesc(c);
%     hold on
%     title('Item Covariance Matrix') 
    
    wth_c=c.*within_class;
    btw_c=c.*between_class;
    
    % compute variance 
    wth_var = var(c(within_class==1));
    
    c_ratio(i)=nanmean(btw_c(:))-nanmean(wth_c(:));
    
    BTW_c(i)=nanmean(btw_c(:));
    WTH_c(i)=nanmean(wth_c(:));
    
    BTW_var(i)=var(c(between_class==1));
    WTH_var(i)=var(c(within_class==1));

    BTW_pdist(i)=mean(pdist(tril(c(between_class==1),-1)));
    WTH_pdist(i)=mean(pdist(tril(c(within_class==1),-1)));
    
    fprintf(strcat(num2str(i),'\n'));
end 

%% pdist
pd = median(pdist((c(between_class==1))));
pd2 = median(pdist((c(within_class==1))));

pd2tril = mean(pdist(tril(c(within_class==1),-1)));

%%
figure;
surf(Beta,Sigma,WTH_c);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('mean cov_{wth}')
set(gca,'fontsize',16)
%
figure;
surf(Beta,Sigma,BTW_c);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('mean cov_{btw}')
set(gca,'fontsize',16)
%
figure;
surf(Beta,Sigma,BTW_c./WTH_c);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('mean cov_{btw}/cov_{wth}')
set(gca,'fontsize',16)

% within var
figure;
surf(Beta,Sigma,WTH_var);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('var_{wth}')
set(gca,'fontsize',16)

% between var
figure;
surf(Beta,Sigma,BTW_var);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('var_{wth}')
set(gca,'fontsize',16)

% ratio var
figure;
surf(Beta,Sigma,BTW_var./WTH_var);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('var_{btw} / var_{wth}')
set(gca,'fontsize',16)

% pdist btw
figure;
surf(Beta,Sigma,BTW_pdist);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('(euclidean dist)_{btw}')
set(gca,'fontsize',16)

% pdist wth
figure;
surf(Beta,Sigma,WTH_pdist);shg
view([143.2000,18.4000]);shg
xlabel('beta')
ylabel('sigma')
zlabel('(euclidean dist)_{wth}')
set(gca,'fontsize',16)

%% 
ops=neural_manifold_create_synth_data_cholesky_method_gt('n_class',20,'n_feat',100,'exm_per_class',10,'beta',0.022,'sigma',0.833);
c=cov(transpose(ops.data));
figure;
imagesc(c)
