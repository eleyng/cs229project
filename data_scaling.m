function [data_unit_var] = data_scaling(data)
%%
%   Take data (with features in columns) and shift + normalize to create
%   new data matrix with zero mean and unit variance
%
    data_shift = bsxfun(@minus, data, mean(data));
    data_unit_var = bsxfun(@rdivide, data_shift, sqrt(var(data)));
end 