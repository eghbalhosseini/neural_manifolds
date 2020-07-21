function [] = plot_decomp(data, varargin)
p=inputParser();
%addParameter(p, 'data', data);
addParameter(p, 'labels', []);
% addParameter(p, 'n_feat', 28*28);
% addParameter(p, 'beta', 0.01);
% addParameter(p, 'structure', 'partition'); % options : partition , tree
% addParameter(p, 'sigma', 1.5);
% addParameter(p,'norm',true);
addParameter(p, 'save_path', '~/');
parse(p, varargin{:});

c=cov(data');
figure;imagesc(c);
hold on
title('Item Covariance Matrix') %Covariance matrix of the dataset
if p.Results.labels ~= []
    xticks(1:length(labels)); xticklabels(animals.names);xtickangle(55);
    hold on
    yticks(1:length(labels)); yticklabels(animals.names);
end

%% SVD
[U,S,V] = svd(c);
figure;imagesc(U');hold on;
title('Input-analyzing Singular Vectors')

%%
Vt=V';
figure;imagesc(U(:,1:3)*S(1:3,1:3)*Vt(1:3,:)); hold on;
title('Dataset explained by first three singular vectors')


figure;imagesc(U(:,1)*S(1,1)*Vt(1,:)); hold on;
title('Dataset explained by first singular vector')

% figure;imagesc(U(:,3)*S(3,3)*Vt(3,:))

%% EIG
[eigVecs, eigVals] = eig(c);

% Flip the eigenvectors so I can index the sorted eigenvectors in terms of
% variance explained
eigVecs = fliplr(eigVecs);

sorted_eigVals = sort(diag(eigVals),'descend'); % Sort eigenvalues

% Make Scree plot
var_per_PC = sorted_eigVals/sum(sorted_eigVals);
cum_var_per_PC = cumsum(sorted_eigVals/sum(sorted_eigVals));

figure;
plot(1:length(sorted_eigVals), var_per_PC)
hold on
plot(1:length(sorted_eigVals), cum_var_per_PC)
hold on
legend('Variance explained per PC', 'Cumulative variance explained per PC')
xticks(1:20:length(sorted_eigVals))
xlim([0 length(sorted_eigVals)])
ylim([0 1])
xlabel('Principal component (PC) number')
ylabel('Variance explained')
title({'Variance and cumulative variance explained by each PC'})

%% MDS
DM = pdist(data', 'euclidean');
DM = squareform(DM);
cmd = cmdscale(DM);

figure;plot(cmd(:,1),cmd(:,2),'.')

end

