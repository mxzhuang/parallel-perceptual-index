#include "modules.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include "matlab_compat.h"
#include "parallel.h"

namespace {
// ============================================================================
// M3: patch SVD (svd(im2col(im, [5 5], 'distinct')))
// ============================================================================
constexpr int kPatch = 5;
constexpr int kDim = kPatch * kPatch;  // 25 values per patch, 25 singular values per scale
// Rows per block of the tall-skinny QR. A block (1024 x 25 doubles, 200 KB) fits in a
// private cache, and the serial reduction of the stacked factors stays near 25/1024 of
// the work of the blocks.
constexpr long kBlockRows = 1024;

PI_NOINLINE Vec patch_svd(const Mat& im) { return mc::singular_values_wide(mc::im2col_distinct(im, kPatch, kPatch)); }

// Householder QR in place, with the arithmetic of mc::singular_values_wide: on return the
// upper triangle of b holds R. The columns form a chain (column k is reduced with the
// reflections of columns 0..k-1), so this loop stays serial; parallelism is across blocks.
void householder_qr(Mat& b) {
    const int n = b.rows, m = b.cols;
    const int k_end = std::min(m, n);
    Vec v(n);  // one scratch vector per factorization, not one per column
    for (int k = 0; k < k_end; ++k) {
        double norm = 0;
        for (int i = k; i < n; ++i) norm += b(i, k) * b(i, k);
        norm = std::sqrt(norm);
        if (norm == 0) continue;
        const double alpha = b(k, k) > 0 ? -norm : norm;
        for (int i = k; i < n; ++i) v[i - k] = b(i, k);
        v[0] -= alpha;
        double vn = 0;
        for (int i = 0; i < n - k; ++i) vn += v[i] * v[i];
        if (vn == 0) continue;
        for (int j = k; j < m; ++j) {
            double dot = 0;
            for (int i = k; i < n; ++i) dot += v[i - k] * b(i, j);
            const double f = 2 * dot / vn;
            for (int i = k; i < n; ++i) b(i, j) -= f * v[i - k];
        }
    }
}

// Rows [lo, hi) of the transposed patch matrix, taken straight from the image: row p is
// the 5x5 patch number p (column-major patch order, zero beyond the image border).
Mat patch_rows(const Mat& im, int nbr, long lo, long hi) {
    Mat b(static_cast<int>(hi - lo), kDim);
    for (int cc = 0; cc < kPatch; ++cc)
        for (int rr = 0; rr < kPatch; ++rr) {
            double* col = &b.d[static_cast<size_t>(cc * kPatch + rr) * b.rows];
            for (long p = lo; p < hi; ++p) {
                const int r = static_cast<int>(p % nbr) * kPatch + rr;
                const int c = static_cast<int>(p / nbr) * kPatch + cc;
                col[p - lo] = (r < im.rows && c < im.cols) ? im(r, c) : 0.0;
            }
        }
    return b;
}

// Singular values of the upper-triangular factor held in the first r rows of b, by
// one-sided Jacobi (Hestenes) on its columns, as in mc::singular_values_wide.
Vec jacobi_singular_values(const Mat& b, int r) {
    const int m = b.cols;
    Mat R(r, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i <= std::min(j, r - 1); ++i) R(i, j) = b(i, j);
    for (int sweep = 0; sweep < 60; ++sweep) {
        bool rotated = false;
        for (int p = 0; p < m - 1; ++p)
            for (int q = p + 1; q < m; ++q) {
                double alpha = 0, beta = 0, gamma = 0;
                for (int i = 0; i < r; ++i) {
                    alpha += R(i, p) * R(i, p);
                    beta += R(i, q) * R(i, q);
                    gamma += R(i, p) * R(i, q);
                }
                if (gamma == 0 || std::fabs(gamma) <= 1e-15 * std::sqrt(alpha * beta)) continue;
                rotated = true;
                const double zeta = (beta - alpha) / (2 * gamma);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                const double cs = 1 / std::sqrt(1 + t * t), sn = cs * t;
                for (int i = 0; i < r; ++i) {
                    const double x = R(i, p), y = R(i, q);
                    R(i, p) = cs * x - sn * y;
                    R(i, q) = sn * x + cs * y;
                }
            }
        if (!rotated) break;
    }
    Vec sv(m);
    for (int j = 0; j < m; ++j) {
        double s = 0;
        for (int i = 0; i < r; ++i) s += R(i, j) * R(i, j);
        sv[j] = std::sqrt(s);
    }
    std::sort(sv.begin(), sv.end(), std::greater<double>());
    sv.resize(r);
    return sv;
}

// Tall-skinny QR of the N x 25 transposed patch matrix A' (N = number of patches). The
// singular values of A are those of the R factor of A', and
//     A' = [A1; A2; ...] = diag(Q1, Q2, ...) * [R1; R2; ...],
// so the row blocks can be factored independently and only their 25x25 factors have to be
// reduced together. The full im2col matrix and its transpose are never stored.
PI_NOINLINE Vec patch_svd_tsqr(const Mat& im, const Config& cfg) {
    const int nbr = (im.rows + kPatch - 1) / kPatch, nbc = (im.cols + kPatch - 1) / kPatch;
    const long n = static_cast<long>(nbr) * nbc;
    // The block grid depends on the number of patches only, never on the thread count or
    // the schedule: the same blocks are factored and stacked in the same order in every
    // run, which keeps the result bit-identical for any number of threads.
    const long nblk = std::max(1L, n / kBlockRows);
    if (nblk == 1) {  // also covers scales with fewer than 25 patches
        Mat b = patch_rows(im, nbr, 0, n);
        householder_qr(b);
        return jacobi_singular_values(b, static_cast<int>(std::min<long>(kDim, n)));
    }

    // Block i owns the contiguous slice [i * 625, (i + 1) * 625) of `factors`; its matrix
    // and scratch are allocated once per block inside the chunk. Blocks have equal size,
    // so the default static partition is balanced.
    Vec factors(static_cast<size_t>(nblk) * kDim * kDim);
    par_for(cfg, nblk, Sched{}, [&](long blo, long bhi) {
        for (long i = blo; i < bhi; ++i) {
            long lo, hi;
            chunk_bounds(n, nblk, i, lo, hi);
            Mat b = patch_rows(im, nbr, lo, hi);
            householder_qr(b);
            double* r = &factors[static_cast<size_t>(i) * kDim * kDim];
            for (int j = 0; j < kDim; ++j)
                for (int k = 0; k <= j; ++k) r[j * kDim + k] = b(k, j);
        }
    });

    // Merge once, after the join, in block order: stack the factors and reduce them with
    // one more QR. This step is serial but small (25 * nblk rows).
    Mat stacked(static_cast<int>(nblk) * kDim, kDim);
    for (long i = 0; i < nblk; ++i) {
        const double* r = &factors[static_cast<size_t>(i) * kDim * kDim];
        for (int j = 0; j < kDim; ++j)
            for (int k = 0; k <= j; ++k) stacked(static_cast<int>(i) * kDim + k, j) = r[j * kDim + k];
    }
    householder_qr(stacked);
    return jacobi_singular_values(stacked, kDim);
}
}  // namespace

Vec m3_patch_svd(const SpatialPyramid& sp, const Config& cfg) {
    // Each scale fills its own vector; they are concatenated in scale order afterwards.
    Vec sv[3];
    par_outer(cfg, 3, [&](int s) { sv[s] = cfg.designed() ? patch_svd_tsqr(sp.im[s], cfg) : patch_svd(sp.im[s]); });
    Vec f3;
    for (const Vec& v : sv) f3.insert(f3.end(), v.begin(), v.end());
    return f3;
}
