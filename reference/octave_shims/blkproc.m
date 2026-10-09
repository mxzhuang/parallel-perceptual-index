function b = blkproc(a, blk, border, fun)
% MATLAB-compatible blkproc(A,[m n],[mb nb],fun): zero padding to a multiple of the
% block size at the bottom/right, plus a zero border of [mb nb] around the image.
m = blk(1); n = blk(2); mb = border(1); nb = border(2);
[ma, na] = size(a);
mpad = rem(ma, m); if mpad > 0, mpad = m - mpad; end
npad = rem(na, n); if npad > 0, npad = n - npad; end
aa = zeros(ma + mpad + 2*mb, na + npad + 2*nb);
aa(mb+1:mb+ma, nb+1:nb+na) = a;
mblocks = (ma + mpad) / m; nblocks = (na + npad) / n;
arows = 1:(m + 2*mb); acols = 1:(n + 2*nb);
first = feval(fun, aa(arows, acols));
[p, q] = size(first);
b = zeros(p*mblocks, q*nblocks);
rows = 1:p; cols = 1:q;
for i = 0:mblocks-1
  for j = 0:nblocks-1
    b(i*p+rows, j*q+cols) = feval(fun, aa(i*m+arows, j*n+acols));
  end
end
end
