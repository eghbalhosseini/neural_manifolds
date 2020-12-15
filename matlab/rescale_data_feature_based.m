% adopted from : Lake, B. M., Lawrence, N. D., and Tenenbaum, J. B. (2018). The emergence of organizing structure in conceptual representation. 
% Cognitive Science, 42(S3), 809-832.
% Transform data such that the mean entry is 0 and the max entry in the cov. matrix is 1
%
% Data should be objects x features
function [data,cov_new] = rescale_data_feature_based(data)
    [n,m] = size(data);
    % demean
    data = data - repmat(mean(data,1),n,1);
    % rescale 
    Y_org = calc_cov(data);
    mx = max(Y_org(:));
    data = data ./ sqrt(mx);
    cov_new = calc_cov(data);
    assert(aeq(max(cov_new(:)),1));
    assert(aeq(mean(data(:)),0));
end