function [] = plot_tree_decomp(data, varargin)
p=inputParser();
addParameter(p, 'labels', []);
addParameter(p, 'save_path', '~/');
addParameter(p, 'plot_str', 'X.pdf');
parse(p, varargin{:});
ops = p.Results;


%% 
if ~exist(ops.save_path, 'dir')
   mkdir(ops.save_path)
end

%%
c = cov(data');
figure;imagesc(c);
hold on
colorbar()
hold on
title('Item Covariance Matrix'); hold on;
saveas(gcf, strcat(ops.save_path, 'cov_', ops.plot_str));
% Covariance matrix of the dataset
if p.Results.labels ~= []
    xticks(1:length(labels)); xticklabels(animals.names);xtickangle(55);
    hold on
    yticks(1:length(labels)); yticklabels(animals.names);
end

% %% SVD
% [U,S,V] = svd(c); % the singular values are sorted
% Ut = U';
% 
% %%
% figure;imagesc(Ut(1:5,:));hold on;colorbar();
% title('Input-Analyzing Singular Vectors'); hold on;
% saveas(gcf, strcat(ops.save_path, 'input_sing_vectors_', ops.plot_str));
% 
% %%
% Vt=V';
% 
% figure;imagesc(U(:,1)*S(1,1)*Vt(1,:)); hold on;
% colorbar(); hold on;
% title('Dataset explained by first singular vector'); hold on;
% saveas(gcf, strcat(ops.save_path, 'sing_vector1_', ops.plot_str));
% 
% figure;imagesc(U(:,2)*S(2,2)*Vt(2,:)); hold on;
% colorbar(); hold on;
% title('Dataset explained by first two singular vectors'); hold on;
% saveas(gcf, strcat(ops.save_path, 'sing_vector2_', ops.plot_str));
% 
% figure;imagesc(U(:,1:3)*S(1:3,1:3)*Vt(1:3,:)); hold on;
% colorbar(); hold on;
% title('Dataset explained by first three singular vectors'); hold on;
% saveas(gcf, strcat(ops.save_path, 'sing_vector3_', ops.plot_str));
% 
% figure;imagesc(U(:,1:4)*S(1:4,1:4)*Vt(1:4,:)); hold on;
% colorbar(); hold on;
% title('Dataset explained by first four singular vectors'); hold on;
% saveas(gcf, strcat(ops.save_path, 'sing_vector4_', ops.plot_str));
% 
% figure;imagesc(U(:,1:5)*S(1:5,1:5)*Vt(1:5,:)); hold on;
% colorbar(); hold on;
% title('Dataset explained by first five singular vectors'); hold on;
% saveas(gcf, strcat(ops.save_path, 'sing_vector5_', ops.plot_str));
% 
% 

% figure;imagesc(U(:,3)*S(3,3)*Vt(3,:))

% cd ..\..
end

