function B = imresize(A, s)
% MATLAB-compatible bicubic imresize (antialiasing on), 2-D double input only.
A = double(A);
in = size(A);
if numel(s) == 1
  scale = [s s];
  out = ceil(scale .* in);
else
  out = s;
  scale = out ./ in;
end
[~, order] = sort(scale);
B = A;
for d = order
  [w, idx] = contributions(size(B, d), out(d), scale(d));
  if d == 1
    C = zeros(out(1), size(B, 2));
    for k = 1:size(w, 2)
      C = C + w(:, k) .* B(idx(:, k), :);
    end
  else
    C = zeros(size(B, 1), out(2));
    for k = 1:size(w, 2)
      C = C + w(:, k)' .* B(:, idx(:, k));
    end
  end
  B = C;
end
end

function f = cubic(x)
ax = abs(x); ax2 = ax.^2; ax3 = ax.^3;
f = (1.5*ax3 - 2.5*ax2 + 1) .* (ax <= 1) + ...
    (-0.5*ax3 + 2.5*ax2 - 4*ax + 2) .* ((1 < ax) & (ax <= 2));
end

function [weights, indices] = contributions(in_len, out_len, scale)
kw = 4;
if scale < 1
  h = @(x) scale * cubic(scale * x);
  kw = kw / scale;
else
  h = @cubic;
end
x = (1:out_len)';
u = x / scale + 0.5 * (1 - 1 / scale);
left = floor(u - kw / 2);
P = ceil(kw) + 2;
indices = left + (0:P-1);
weights = h(u - indices);
weights = weights ./ sum(weights, 2);
aux = [1:in_len, in_len:-1:1];
indices = aux(mod(indices - 1, numel(aux)) + 1);
kill = find(~any(weights, 1));
weights(:, kill) = [];
indices(:, kill) = [];
end
