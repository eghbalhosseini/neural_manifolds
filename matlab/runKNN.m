function M = runKNN(varargin)
p=inputParser();
addParameter(p, 'root_dir', '/mindhive/evlab/u/Shared/Greta_Eghbal_manifolds/');
addParameter(p, 'analyze_identifier', 'knn-k=100-dist_metric=euclidean-num_subsamples=100');
addParameter(p, 'extraction_identifier', 'mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5');
addParameter(p, 'model_identifier', 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=2.5_nfeat=3072-train_test-fixed');
addParameter(p, 'training_folder', 'epochs-10_batch-32_lr-0.001_momentum-0.5_init-gaussian_std-0.0001');
addParameter(p, 'layer', 'layer_3_Linear');
addParameter(p, 'dist_metric', 'euclidean');
addParameter(p, 'save_fig', true);
addParameter(p, 'k', 100);
addParameter(p, 'num_subsamples', 100);

parse(p, varargin{:});
params = p.Results;
params_out = params;

disp(strcat('Model identifier: ', params.model_identifier))
disp(strcat('Layer: ', params.layer))
disp(strcat('Training folder: ', params.training_folder))
disp(strcat('Extraction identifier: ', params.extraction_identifier))

%% Figure specs
set(groot, ...
'DefaultFigureColor', 'w', ...
'DefaultAxesLineWidth', 0.5, ...
'DefaultAxesXColor', 'k', ...
'DefaultAxesYColor', 'k', ...
'DefaultAxesFontUnits', 'points', ...
'DefaultAxesFontSize', 10, ...
'DefaultAxesFontName', 'Helvetica', ...
'DefaultLineLineWidth', 1, ...
'DefaultTextFontUnits', 'Points', ...
'DefaultTextFontSize', 10, ...
'DefaultTextFontName', 'Helvetica', ...
'DefaultAxesBox', 'off', ...
'DefaultAxesTickLength', [0.01 0.015]);
set(groot, 'DefaultAxesTickDir', 'out');
set(groot, 'DefaultAxesTickDirMode', 'manual');
set(gcf,'color','w');

%% Directories
% tmp = matlab.desktop.editor.getActive;
% cd(fileparts(tmp.Filename));

addpath(strcat('/om/user/gretatu/neural_manifolds/matlab/utils/'))
addpath(strcat('/om/user/ehoseini/neural_manifolds/matlab/utils/'))

% Load the generated mat files, session of interest: (input, the model identifier)

%% Manual input
params.root_dir = '/Users/gt/Documents/GitHub/neural_manifolds/local/knn_tests/'
params.model_identifier = 'NN-tree_nclass=64_nobj=64000_nhier=6_beta=0.0_sigma=0.5_nfeat=936-train_test-fixed'
params.layer = 'layer_3_Linear'
params.analyze_identifier = 'knn-k=100-dist_metric=euclidean-num_subsamples=100'
params.k = 100
params.num_subsamples = 100
params.save_fig = false
params.dist_metric = 'euclidean'
params.training_folder = 'epochs-10_batch-32_lr-0.001_momentum-0.5_init-gaussian_std-0.0001'
params_out = params;
params.extraction_identifier = 'mftma-exm_per_class=50-proj=False-rand=True-kappa=1e-08-n_t=300-n_rep=5'

addpath(strcat('/Users/gt/Documents/GitHub/neural_manifolds/matlab/utils/'))

%%
dataDir = strcat(params.root_dir, '/extracted/', params.extraction_identifier, filesep);
analyzeDir = strcat(params.root_dir, 'analyze/', params.analyze_identifier, filesep, params.model_identifier, filesep, params.training_folder, filesep);
resultDir = strcat(params.root_dir, 'result/', params.analyze_identifier, filesep, params.model_identifier, filesep, params.training_folder, filesep);

%Make directories according to analyzeID
if ~exist(analyzeDir, 'dir')
   mkdir(analyzeDir)
end

if ~exist(resultDir, 'dir')
   mkdir(resultDir)
end


KNN_files = dir(strcat(dataDir, params.model_identifier, filesep, params.training_folder, filesep, '*', params.layer, '_extracted.mat'));
disp('Found KNN files!')

order = cellfun(@(x) str2num(x(1:4)), {KNN_files.name}, 'UniformOutput', false);
assert(issorted(cell2mat(order)), 'Files not ordered correctly!')

%% Load files
KNN_data = arrayfun(@(x) {strcat(KNN_files(x).folder, filesep, KNN_files(x).name)}, 1:length(KNN_files));
disp(strcat('Searching for KNN files in: ', (strcat(dataDir, params.model_identifier, filesep, params.training_folder))))
assert(~isempty(KNN_data), 'KNN files empty - no files found!')

file = load(KNN_data{1});
file = file.activation;

% Iterate over all hierarchies in the file 
nhier_idx = strfind(params.model_identifier,'nhier');
nhier_level = str2num(params.model_identifier(nhier_idx + length('nhier') + 1));

% Generate cell for saving the maps of results
cell_map = {};
% Generate hier field names and structures in a list
hier_field_names = [];
hier_field_structs = [];

% Save meanTimeNormNN and stdTimeNorm NN across all hierarchies
meanTimeNormNN_all = zeros(nhier_level, length(KNN_data));
stdTimeNormNN_all = zeros(nhier_level, length(KNN_data));

% hier_level = 3 %  TEST OUTCOMMENT
for hier_level = 1:nhier_level
disp(strcat('Hierarchy level: ', string(hier_level)))
tic

% Load sample file to get correct dimensions for each hierarchy
sample_file = file.projection_results{1, hier_level}.( params.layer );

data_size = size(sample_file);
num_classes = data_size(1);
num_features = data_size(2);
num_examples = data_size(3);

%% Assert that subsampling across time is possible 
assert(params.num_subsamples <= num_classes * num_examples, 'Too many subsamples specified')
assert(params.num_subsamples >= num_classes, 'Number of subsamples has to be equal to or larger than number of classes')

%% Iterate over all files

data = [];
targets = [];
epoch = [];
subEpoch = [];
testAcc = [];
trainAcc = [];

toc
disp('Start subsampling KNN data files')
tic
for i = 1:length(KNN_data)
    file = load(KNN_data{i});
    file = file.activation;
    f = file.projection_results{1, hier_level}.( params.layer );
    
    % Subsample and construct a data matrix 
    f_perm = permute(f, [3 1 2]);
    f_res = reshape(f_perm, [num_classes*num_examples, num_features]);
    % I.e. concatenated according to: rows = num samples per class, e.g. if
    % num_examples = 20, then the first 20 rows correspond to all samples from
    % class 1
    
    % Cov matrix:
    % figure;imshow(cov(f_res'))
    
    % Generate targets
    target = repelem([1:num_classes], num_examples)';

    % Make sure at least one sample per class per time point
    sub = randsample(num_examples, num_classes, true);
    % Add indices according to where to sample from, taking num_examples
    % into account:
    add = linspace(0, num_examples*num_classes, num_classes+1);
    add_array = add(1:num_classes);

    idx_cat = sub' + add_array; % Indices for subsampling per category

    % Add more indices
    if params.num_subsamples ~= num_classes
        num_remaining_sub = params.num_subsamples - num_classes;
        draw = [1:num_examples*num_classes]; % possible indices to draw from
        draw_array = setdiff(draw, idx_cat); % subtract the ones already used for the category requirement

        % Now draw from this list, to avoid sampling the same data points 
        more_idx = randsample(draw_array, num_remaining_sub, false); % without replacement 

    end

    final_idx = horzcat(idx_cat, more_idx)';

    assert(length(final_idx) == params.num_subsamples, 'Subsampling index does not match')

    % Subsample
    subsampled_data = f_res(final_idx,:); % Checked that it correspond to the first idx in the sub list
    subsampled_target = target(final_idx);
    
    % get batch/epoch 
    batchidx_cell = file.batchidx;
    batchs = repmat(batchidx_cell, params.num_subsamples, 1);
    
    epoch_cell = file.epoch;
    epochs = repmat(epoch_cell, params.num_subsamples, 1);
    
    % Get accuracies
    test_a = file.test_acc;
    train_a = file.train_acc;

    
    % Append
    data = [data; subsampled_data];
    epoch = double([epoch; epochs]);
    subEpoch = double([subEpoch; batchs]);
    targets = [targets; subsampled_target];
    testAcc = [testAcc; test_a];
    trainAcc = [trainAcc; train_a];

end
toc
% Find log interval:
logInt = abs(subEpoch(params.num_subsamples) - subEpoch(params.num_subsamples*2));
productionTime = (1:length(epoch))'; 
time = [1:length(KNN_files)]';
relTime = productionTime./max(productionTime);

%% %%%%%%%%%% SAVE DIRS, COLORS, AXES LABELS %%%%%%%%%%%%%%

% Save directories
saveStrResult = strcat(params.model_identifier,'_',params.layer,'_hier_',num2str(hier_level),'_numSubsamples_',num2str(params.num_subsamples),'_k_',num2str(params.k),'.pdf');
saveStrAnalyze = strcat(params.model_identifier,'_',params.layer,'_numSubsamples_',num2str(params.num_subsamples),'_k_',num2str(params.k),'.mat');

% Colors
colorsEpoch = magma(max(epoch)+1); % 3 colors
colorsEpoch = colorsEpoch(1:end-1, :);

colorsTarget = magma(max(num_classes+1));
colorsTarget = colorsTarget(1:end-1, :);

colorsTargetViridis = viridis(max(num_classes));

colorsMeanTimeNormNN_all = magma(nhier_level+1); 
colorsMeanTimeNormNN_all = colorsMeanTimeNormNN_all(1:end-1, :);

%% Make axes labels
subEpochSwitch = diff(subEpoch);
subEpochSwitch = [0;subEpochSwitch];

subepochLoc = arrayfun(@(x) find(subEpoch==x), unique(subEpoch), 'uniformoutput',false); % finds the unique subEpochs
epochLoc = arrayfun(@(x) find(epoch==x), [1:max(epoch)],'uniformoutput',false); % Locates the epochs
epochSwitch = (subEpochSwitch==(min(subEpochSwitch)));

p = subEpochSwitch.*double(epochSwitch) - max(subEpochSwitch).*double(epochSwitch); % account for that epochSwitch only contains 1, and not log interval
pp = [cumsum(-p)]; 
incrTrial = subEpoch + pp;

uniqueSubEpochBatches = arrayfun(@(x) find(incrTrial==x), unique(incrTrial), 'uniformoutput',false); % finds unique number of subEpoch batches
incrSubEpochTrial = cell2mat(cellfun(@(x) min(x), uniqueSubEpochBatches, 'uniformoutput',false));
incrSubEpoch = subEpoch((incrSubEpochTrial));

targetLoc = arrayfun(@(x) find(targets==x), [1:max(target)],'uniformoutput',false); % Locates the targets

%% Permutation across time
% Proof of concept that KNN can handle data that is not organized correctly
% data1 = data;
% rand_idx = randperm(length(data1));
% data_permute = data1(rand_idx, :); % if rand_idx(1) is 71, then row71 in the real data is now row 1

%% %%%%%%%%%% ANALYSES %%%%%%%%%%%%%%

%% Test/train accuracy

% Test acc for each subsample - if overlay on the other plots 
testAccSubsample = repmat(testAcc, 1, params.num_subsamples);
testAccSubsamples = reshape(testAccSubsample', [], 1);

% Train acc for each subsample - if overlay on the other plots
trainAccSubsample = repmat(trainAcc, 1, params.num_subsamples);
trainAccSubsamples = reshape(trainAccSubsample', [], 1);

if params.save_fig
    % Plot test accuracy
    figure;
    ax=axes()
    hold on
    arrayfun(@(i) plot(productionTime(epochLoc{i})', testAccSubsamples(epochLoc{i})', 'Linewidth', 1, 'Color', colorsEpoch(i,:)), [1:max(epoch)]);
    hold on
    ylabel('Test accuracy')
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(resultDir,'testAcc_',saveStrResult));
    
    % If label using subepoch indexing:
%     set(gca,'XTick',downsample(incrSubEpochTrial,15))
%     set(gca,'XTickLabel',downsample(incrSubEpoch,15))
%     xlabel('Subepoch')

    % If label using epoch indexing:
%     set(gca,'XTick',downsample(productionTime, round(size(productionTime,1)/max(epoch))))
%     set(gca,'XTickLabel',downsample(epoch, round(size(productionTime,1)/max(epoch))))
end 

%% Vector norm
dataNorm = vecnorm(data');

% Epoch coloring
if params.save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(epochLoc{i})', dataNorm(epochLoc{i})', 2, colorsEpoch(i,:), 'filled', 'o'), [1:max(epoch)])
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    ylabel('Norm')
    axis tight
    saveas(gcf, strcat(resultDir,'norm_epochColor_',saveStrResult));
end 

% Target coloring
if params.save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', dataNorm(targetLoc{i})', 8, colorsTarget(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    ylabel('Norm')
    axis tight
    saveas(gcf, strcat(resultDir,'norm_targetColor_',saveStrResult));
end 

%% Vector norm - meaned across samples
y = reshape(dataNorm, params.num_subsamples, length(KNN_files));
meanTimeDataNorm = mean(y, 1);

%% Nearest neighbors
disp('Computing KNN')
tic
NNids_self = knnsearch(data, data, 'K', params.k, 'Distance', params.dist_metric); 
NNids_self = NNids_self./max(NNids_self(:,1)); % normalized NNids
NNids = NNids_self(:, 2:end); 
toc

% Null - if the network did not perform any operation, how should the KNN
% representation look
nullNNids = repmat(NNids_self(:,1),1,200);
% figure;imagesc(nullNNids)

disp('Starting KNN plots')
tic
if params.save_fig
    figure;
    imagesc(NNids_self)
    hold on
    xlabel('K nearest neighbors')
    set(gca,'YTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'YTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    ylabel('Relative time in training')
    colorbar()
    axis tight
    saveas(gcf, strcat(resultDir,'nearestNeighbors_',saveStrResult));
end  

y1 = reshape(NNids, params.num_subsamples, length(KNN_files), params.k-1); % num subsamples x num points in time x num k-1
meanTimeNNids = squeeze(mean(y1, 1));
% figure;imagesc(meanTimeNNids)

%% Compute norm of neighbors
normNN = zeros(params.k - 1, length(epoch));

for i=2:params.k
    normNN(i-1, :)=(abs(NNids_self(:,1) - NNids_self(:,i)));%.^2;
end

meanNormNN = mean(normNN,1);
stdNormNN = std(normNN,1);

%% Compute norm of neighbors - null
normNN_null = zeros(params.k - 1, length(epoch));

for i=2:params.k
    normNN_null(i-1, :)=(abs(nullNNids(:,1) - nullNNids(:,i)));%.^2;
end

meanNormNN_null = mean(normNN_null,1);
stdNormNN_null = std(normNN_null,1);

y2 = reshape(meanNormNN_null, params.num_subsamples, length(KNN_files));
meanTimeNormNN_null = mean(y2, 1);

y3 = reshape(stdNormNN_null, params.num_subsamples, length(KNN_files));
meanTimeStdNormNN_null = mean(y3, 1);

%% Mean vector norms over samples at the same time 
y2 = reshape(meanNormNN, params.num_subsamples, length(KNN_files));
meanTimeNormNN = mean(y2, 1);
stdTimeNormNN = std(y2, 1);

% std over the KNN matrix
y3 = reshape(stdNormNN, params.num_subsamples, length(KNN_files));
meanTimeStdNormNN = mean(y3, 1);

%% Plot vector norm - according to epochs colors
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.

if params.save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(epochLoc{i})', meanNormNN(epochLoc{i})', 1, colorsEpoch(i,:), 'filled', 'o'), [1:max(epoch)])
    ylabel('Neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(resultDir,'meanNormNN_',saveStrResult));
end


%% Plot vector norm - according to data point colors - OBS HEAVY PLOT
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
% colorsNormNN = magma(length(meanNormNN)); % for the heavy plot
% if params.save_fig
%     figure;
%     hold on
%     arrayfun(@(i) scatter(productionTime(i)', meanNormNN(i)', 10, colorsNormNN(i,:), 'filled', 'o'), [1:length(meanNormNN)])
%     ylabel('Neighbor distance') % (unit: training time)
%     hold on
%     set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
%     set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
%     xlabel('Relative time in training')
%     axis tight
%     saveas(gcf, strcat(pwd,filesep,'figures',filesep,'meanNormNN_indColors_',saveStr));
% end

%% Plot vector norm - according to data point colors - meaned over time
% Plotting all samples, i.e. if num_samples=60, then 60 samples for that time point. Averaged across neighbors.
colorsMeanTimeNormNN = magma(length(meanTimeNormNN)+1); 
colorsMeanTimeNormNN = colorsMeanTimeNormNN(2:end, :);

if params.save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(time(i)', meanTimeNormNN(i)', 30, colorsMeanTimeNormNN(i,:), 'filled', 'o'), [1:length(meanTimeNormNN)])
    ylabel('Mean neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(time, round(size(time,1)/10)))
    set(gca,'XTickLabel', linspace(0,max(epoch)-1,10))
    % set(gca,'XTickLabel', (max(epochs)))
    xlabel('Epochs')
    axis tight
    saveas(gcf, strcat(resultDir,'meanTimeNormNN_',saveStrResult));
end


%% KNN Vector norm and std in same plot

if params.save_fig
    figure;
    scatter(productionTime', meanNormNN', 1, colorsEpoch(1,:), 'filled', 'o')
    hold on
    scatter(productionTime', stdNormNN', 1, colorsEpoch(2,:), 'filled', 'o')
    hold on
    legend('Mean neighbor distance','Std neighbor distance')
    ylabel('Neighbor distance') 
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(resultDir,'meanStdTimeNormNN_',saveStrResult));
end

%% Target coloring

if params.save_fig
    figure;
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', meanNormNN(targetLoc{i})', 1, colorsTarget(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    arrayfun(@(i) scatter(productionTime(targetLoc{i})', stdNormNN(targetLoc{i})', 1, colorsTargetViridis(i,:), 'filled', 'o'), [1:max(targets)])
    hold on
    %legend('Mean neighbor distance','Std neighbor distance')
    ylabel('Neighbor distance') 
    hold on
    set(gca,'XTick',downsample(productionTime, round(size(relTime,1)/10)))
    set(gca,'XTickLabel',downsample(round(relTime,2), round(size(relTime,1)/10)))
    xlabel('Relative time in training')
    axis tight
    saveas(gcf, strcat(resultDir,'meanStdTimeNormNN_targetColor_',saveStrResult));
end
toc
%% Save variables

%% First across all subsamples, then averaged - pairwise

%% If saving as map structure
keySet = {'hier_level','params', 'targets', 'testAccSubsamples', 'testAcc', 'trainAccSubsamples', 'trainAcc', ...
    'dataNorm', 'meanTimeDataNorm', 'NNids', 'meanTimeNNids', 'meanNormNN', 'meanTimeNormNN', ... 
    'stdNormNN', 'meanTimeStdNormNN'};

valueSet = {hier_level, params_out, targets, testAccSubsamples, testAcc, trainAccSubsamples, trainAcc, ...
    dataNorm, meanTimeDataNorm, NNids, meanTimeNNids, meanNormNN, meanTimeNormNN, ...
     stdNormNN, meanTimeStdNormNN};
 
M = containers.Map(keySet,valueSet,'UniformValues',false);
% Get vals
% M('params')

cell_map{1, hier_level} = M;

% Clear variables for saving the next hierarchy results
clear keySet valueSet M 

%% If saving as struct
% Structs per hierarchy OUTCOMMENTED
hier_struct = struct('hier_level', hier_level, ...
                    'params', params_out, ...
                    'targets', targets, ...
                    'testAccSubsamples', testAccSubsamples, ...
                    'testAcc', testAcc, ...
                    'trainAccSubsamples', trainAccSubsamples, ...
                    'trainAcc', trainAcc, ...
                    'dataNorm', dataNorm, ...
                    'meanTimeDataNorm', meanTimeDataNorm, ...
                    'NNids', NNids, ...
                    'meanTimeNNids', meanTimeNNids, ...
                    'meanNormNN', meanNormNN, ...
                    'meanTimeNormNN', meanTimeNormNN, ...
                    'stdNormNN', stdNormNN, ...
                    'meanTimeStdNormNN', meanTimeStdNormNN)

hier_field_name = strcat('hier_', num2str(hier_level));

% Append to array
hier_field_names = [hier_field_names; hier_field_name];
hier_field_structs = [hier_field_structs; hier_struct];

%% Save meanTimeNormNN and stdTimeNormNN across all hierarchies
meanTimeNormNN_all(hier_level, :) = squeeze(meanTimeNormNN);
stdTimeNormNN_all(hier_level, :) = squeeze(stdTimeNormNN);

%% Clear variables for saving the next hierarchy results
clear hier_field_name hier_field_struct 

end % End hierarchy loop

% Structs of structs
hier_field_names_cell = num2cell(hier_field_names);
hier_field_structs_cell = num2cell(hier_field_structs);

super_struct = cell2struct(hier_field_structs_cell, hier_field_names);

disp(strcat('Finished hierarchy loop - saving file in: ', saveStrAnalyze))

cd(analyzeDir)
% save(saveStrAnalyze, 'cell_map');
save(saveStrAnalyze, 'super_struct','-v7.3');

%% Make meanTimeNormNN plot across all hierarchies

if params.save_fig
    figure;
    
    for i=1:nhier_level
        plot(time, meanTimeNormNN_all(i,:), 'Color', colorsMeanTimeNormNN_all(i,:));hold on;
        plot(time, stdTimeNormNN_all(i, :), 'Color', 'black')
        hold on
    end

    ylabel('Mean neighbor distance') % (unit: training time)
    hold on
    set(gca,'XTick',downsample(time, round(size(time,1)/10)))
    set(gca,'XTickLabel', linspace(0,max(epoch)-1,10))
    xlabel('Epochs');
    hold on
    leg = legend('1','2','3','4','5','6', 'Location', 'best');hold on;
    title(leg, 'Hierarchy level')
    axis tight
    saveas(gcf, strcat(resultDir,'allHier_meanTimeNormNN_',saveStrResult));
end


end



