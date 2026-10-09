function c = nancov(x)
% MATLAB nancov default ('complete'): drop rows containing any NaN, normalize by N-1
x = x(~any(isnan(x), 2), :);
n = size(x, 1);
xc = x - mean(x, 1);
c = (xc' * xc) / (n - 1);
end
