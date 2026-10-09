function m = nanmean(x)
% column-wise mean ignoring NaN (MATLAB Statistics Toolbox semantics for matrices)
nans = isnan(x); x(nans) = 0;
m = sum(x, 1) ./ sum(~nans, 1);
end
