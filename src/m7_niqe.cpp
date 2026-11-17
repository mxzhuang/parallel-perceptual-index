#include "modules.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

void NiqeModel::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    char magic[4];
    uint32_t d = 0;
    if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "PINQ", 4) != 0) throw std::runtime_error("bad NIQE model file");
    if (std::fread(&d, 4, 1, f) != 1) throw std::runtime_error("bad NIQE model file");
    mu.resize(d);
    Vec rowmajor(static_cast<size_t>(d) * d);
    if (std::fread(mu.data(), 8, d, f) != d || std::fread(rowmajor.data(), 8, rowmajor.size(), f) != rowmajor.size())
        throw std::runtime_error("NIQE model file truncated");
    std::fclose(f);
    cov = Mat(d, d);
    for (uint32_t r = 0; r < d; ++r)
        for (uint32_t c = 0; c < d; ++c) cov(r, c) = rowmajor[static_cast<size_t>(r) * d + c];
}

namespace {
constexpr int kFeat = 18;

// computefeature.m for one block
// Baseline: AGGD table built once. Faithful: rebuilt on every call, as estimateaggdparam.m does.
void compute_feature(const Mat& block, double* feat, bool faithful) {
    const AggdTable& aggd = aggd_table();
    auto fit = [&](const double* x, size_t n, double& a, double& l, double& r) {
        if (faithful) AggdTable::fit_faithful(x, n, a, l, r);
        else aggd.fit(x, n, a, l, r);
    };
    double alpha, bl, br;
    fit(block.d.data(), block.numel(), alpha, bl, br);
    feat[0] = alpha;
    feat[1] = (bl + br) / 2;
    static const int shifts[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    Vec pair(block.numel());
    for (int s = 0; s < 4; ++s) {
        Mat sh = mc::circshift(block, shifts[s][0], shifts[s][1]);
        for (size_t i = 0; i < pair.size(); ++i) pair[i] = block.d[i] * sh.d[i];
        fit(pair.data(), pair.size(), alpha, bl, br);
        double meanparam = (br - bl) * (std::tgamma(2 / alpha) / std::tgamma(1 / alpha));
        feat[2 + 4 * s + 0] = alpha;
        feat[2 + 4 * s + 1] = meanparam;
        feat[2 + 4 * s + 2] = bl;
        feat[2 + 4 * s + 3] = br;
    }
}
// Patch features of both scales, sequential baseline (and the faithful variant).
void features_baseline(Mat im, const Mat& window, int mb, int nbk, int bs, bool faithful, Mat& feat) {
    for (int scale = 1; scale <= 2; ++scale) {
        Mat sq(im.rows, im.cols);
        for (size_t i = 0; i < im.numel(); ++i) sq.d[i] = im.d[i] * im.d[i];
        const Mat mu = mc::imfilter(im, window, mc::Boundary::Replicate);
        const Mat ex2 = mc::imfilter(sq, window, mc::Boundary::Replicate);
        Mat structdis(im.rows, im.cols), sigma(im.rows, im.cols);
        for (size_t i = 0; i < im.numel(); ++i) {
            sigma.d[i] = std::sqrt(std::fabs(ex2.d[i] - mu.d[i] * mu.d[i]));
            structdis.d[i] = (im.d[i] - mu.d[i]) / (sigma.d[i] + 1);
        }
        // sharpness = blkproc(sigma, [96 96], @computemean): computed by the official code but
        // never used, so only in the faithful variant
        if (faithful && scale == 1) {
            double s = 0;
            for (int j = 0; j < nbk; ++j)
                for (int i = 0; i < mb; ++i) {
                    const Mat blk = mc::crop(sigma, i * bs, j * bs, bs, bs);
                    s += mc::mean(blk.d.data(), blk.numel());
                }
            g_unused_sink = s;
        }
        const int b = bs / scale;
        double f[kFeat];
        for (int j = 0; j < nbk; ++j)
            for (int i = 0; i < mb; ++i) {
                compute_feature(mc::crop(structdis, i * b, j * b, b, b), f, faithful);
                for (int k = 0; k < kFeat; ++k) feat(i + j * mb, (scale - 1) * kFeat + k) = f[k];
            }
        if (scale == 1) im = mc::imresize_scale(im, 0.5);
    }
}

// ---- designed parallel version ------------------------------------------------------
constexpr int kWin = 7;             // side of the Gaussian window
constexpr int kHalf = kWin / 2;     // its centre tap

// a(0:nr-1, 0:nc-1), copied in parallel over column ranges.
Mat crop_top_left(const Mat& a, int nr, int nc, const Config& cfg) {
    Mat o(nr, nc);
    par_for(cfg, nc, Sched{}, [&](long lo, long hi) {
        for (long c = lo; c < hi; ++c)
            std::copy_n(&a.d[static_cast<size_t>(c) * a.rows], nr, &o.d[static_cast<size_t>(c) * nr]);
    });
    return o;
}

// MSCN coefficients (im - mu) / (sigma + 1), with mu and sigma from the Gaussian window
// applied to the image and to its square (replicate boundary).
//
// Every output pixel depends only on the inputs, so the columns are independent: the loops
// hand out contiguous ranges of whole columns (uniform work, static partition), each range
// writes only its own columns and reads im / sq, which no thread modifies. The two filters
// and the normalization are fused, so mu, E[x^2] and sigma stay in registers instead of
// three more full-size matrices, and the phase needs two joins instead of four.
// Taps are accumulated in the order of mc::imfilter (window columns outer, rows inner),
// which keeps every value identical to the baseline.
Mat mscn(const Mat& im, const Mat& window, const Config& cfg) {
    const int R = im.rows, C = im.cols;
    Mat sq(R, C);
    par_for(cfg, C, Sched{}, [&](long lo, long hi) {
        for (size_t i = static_cast<size_t>(lo) * R, e = static_cast<size_t>(hi) * R; i < e; ++i) sq.d[i] = im.d[i] * im.d[i];
    });
    Mat out(R, C);
    par_for(cfg, C, Sched{}, [&](long lo, long hi) {
        const double* acol[kWin];  // the seven source columns of the current output column
        const double* qcol[kWin];
        for (long c = lo; c < hi; ++c) {
            for (int j = 0; j < kWin; ++j) {
                const size_t cj = static_cast<size_t>(std::min<long>(C - 1, std::max<long>(0, c + j - kHalf)));
                acol[j] = &im.d[cj * R];
                qcol[j] = &sq.d[cj * R];
            }
            const double* centre = acol[kHalf];
            double* dst = &out.d[static_cast<size_t>(c) * R];
            for (int r = 0; r < R; ++r) {
                double mu = 0, ex2 = 0;
                if (r >= kHalf && r + kHalf < R) {  // interior rows: no clamping needed
                    for (int j = 0; j < kWin; ++j) {
                        const double* h = &window.d[static_cast<size_t>(j) * kWin];
                        const double* a = acol[j] + (r - kHalf);
                        const double* q = qcol[j] + (r - kHalf);
                        for (int i = 0; i < kWin; ++i) {
                            mu += h[i] * a[i];
                            ex2 += h[i] * q[i];
                        }
                    }
                } else {
                    for (int j = 0; j < kWin; ++j) {
                        const double* h = &window.d[static_cast<size_t>(j) * kWin];
                        for (int i = 0; i < kWin; ++i) {
                            const int ri = std::min(R - 1, std::max(0, r + i - kHalf));
                            mu += h[i] * acol[j][ri];
                            ex2 += h[i] * qcol[j][ri];
                        }
                    }
                }
                const double sigma = std::sqrt(std::fabs(ex2 - mu * mu));
                dst[r] = (centre[r] - mu) / (sigma + 1);
            }
        }
    });
    return out;
}

// computefeature.m for the b x b block of `structdis` starting at (r0, c0), using scratch
// buffers of b * b elements owned by the caller. Same values and summation order as
// compute_feature: the products with the circularly shifted block are formed by indexing
// the block itself instead of materializing circshift copies.
void block_feature(const Mat& structdis, int r0, int c0, int b, double* blk, double* pair, double* feat) {
    const AggdTable& aggd = aggd_table();  // built once, read-only afterwards
    for (int c = 0; c < b; ++c) std::copy_n(&structdis.d[static_cast<size_t>(c0 + c) * structdis.rows + r0], b, blk + static_cast<size_t>(c) * b);
    const size_t n = static_cast<size_t>(b) * b;
    double alpha, bl, br;
    aggd.fit(blk, n, alpha, bl, br);
    feat[0] = alpha;
    feat[1] = (bl + br) / 2;
    static const int shifts[4][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
    for (int s = 0; s < 4; ++s) {
        const int dr = shifts[s][0], dc = shifts[s][1];
        for (int c = 0; c < b; ++c) {
            const double* x = blk + static_cast<size_t>(c) * b;
            const double* y = blk + static_cast<size_t>((c - dc + b) % b) * b;  // shifted column
            double* p = pair + static_cast<size_t>(c) * b;
            if (dr == 0) {
                for (int r = 0; r < b; ++r) p[r] = x[r] * y[r];
            } else {  // dr == 1: row r pairs with row r - 1, row 0 wraps to the last row
                p[0] = x[0] * y[b - 1];
                for (int r = 1; r < b; ++r) p[r] = x[r] * y[r - 1];
            }
        }
        aggd.fit(pair, n, alpha, bl, br);
        const double meanparam = (br - bl) * (std::tgamma(2 / alpha) / std::tgamma(1 / alpha));
        feat[2 + 4 * s + 0] = alpha;
        feat[2 + 4 * s + 1] = meanparam;
        feat[2 + 4 * s + 2] = bl;
        feat[2 + 4 * s + 3] = br;
    }
}

// Patch features of both scales, parallel. Bit-identical to features_baseline for every
// thread count and schedule: no sum is split or reordered anywhere.
void features_designed(const Mat& gray, const Mat& window, int mb, int nbk, int bs, Mat& feat, const Config& cfg) {
    // The second scale needs only the cropped input, not any first-scale result, so both
    // images exist before the scales start and the scales can run side by side.
    Mat im[2];
    im[0] = crop_top_left(gray, mb * bs, nbk * bs, cfg);
    im[1] = mc::imresize_scale(im[0], 0.5, cfg);
    const long nblocks = static_cast<long>(mb) * nbk;
    par_outer(cfg, 2, [&](int s) {
        const Mat structdis = mscn(im[s], window, cfg);
        const int b = bs / (s + 1);
        // Blocks are independent (no loop-carried dependence). Their cost is nearly uniform,
        // so a static partition is the default; the schedule is a switch to measure it.
        par_for(cfg, nblocks, cfg.m7_sched, [&](long lo, long hi) {
            // Scratch is allocated once per chunk and reused for all its blocks; a fresh
            // crop, shift and product matrix per block would have every thread queueing in
            // the memory allocator.
            Vec blk(static_cast<size_t>(b) * b), pair(blk.size());
            double f[kFeat];
            for (long idx = lo; idx < hi; ++idx) {
                const int i = static_cast<int>(idx % mb), j = static_cast<int>(idx / mb);
                block_feature(structdis, i * b, j * b, b, blk.data(), pair.data(), f);
                // A block owns one row of feat. Rows of a column-major matrix interleave in
                // memory, so neighbouring blocks share cache lines; the features are therefore
                // built in the local array and stored once per block, which makes a shared
                // line change hands 18 times per block rather than on every update.
                for (int k = 0; k < kFeat; ++k) feat(static_cast<int>(idx), s * kFeat + k) = f[k];
            }
        }, cfg.load_m7);
    });
}
}  // namespace

PI_NOINLINE double m7_niqe(const Mat& gray, const NiqeModel& model, const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    const int bs = 96;
    const int mb = gray.rows / bs, nbk = gray.cols / bs;
    if (mb == 0 || nbk == 0) return std::nan("");

    Mat window = mc::fspecial_gaussian(7, 7.0 / 6.0);
    double total = 0;
    for (int c = 0; c < 7; ++c) {
        double cs = 0;
        for (int r = 0; r < 7; ++r) cs += window(r, c);
        total += cs;
    }
    for (double& v : window.d) v /= total;

    const int nblocks = mb * nbk;
    Mat feat(nblocks, 2 * kFeat);
    // the faithful variant exists only as a sequential reference
    if (cfg.designed() && !faithful) features_designed(gray, window, mb, nbk, bs, feat, cfg);
    else features_baseline(mc::crop(gray, 0, 0, mb * bs, nbk * bs), window, mb, nbk, bs, faithful, feat);

    // optional: dump the patch features for validation (PI_DEBUG_NIQE=<file>)
    if (const char* dbg = std::getenv("PI_DEBUG_NIQE")) {
        if (FILE* f = std::fopen(dbg, "w")) {
            for (int r = 0; r < nblocks; ++r) {
                for (int c = 0; c < 2 * kFeat; ++c) std::fprintf(f, "%.17g ", feat(r, c));
                std::fprintf(f, "\n");
            }
            std::fclose(f);
        }
    }

    // Model phase: 36 x 36 statistics over at most a few hundred blocks, left serial.
    // nanmean / nancov (complete rows)
    const int d = 2 * kFeat;
    Vec mu_dist(d);
    for (int c = 0; c < d; ++c) {
        double s = 0;
        int n = 0;
        for (int r = 0; r < nblocks; ++r)
            if (!std::isnan(feat(r, c))) {
                s += feat(r, c);
                ++n;
            }
        mu_dist[c] = s / n;
    }
    std::vector<int> rows;
    for (int r = 0; r < nblocks; ++r) {
        bool ok = true;
        for (int c = 0; c < d && ok; ++c) ok = !std::isnan(feat(r, c));
        if (ok) rows.push_back(r);
    }
    const int n = static_cast<int>(rows.size());
    Vec colmean(d, 0.0);
    for (int c = 0; c < d; ++c) {
        double s = 0;
        for (int r : rows) s += feat(r, c);
        colmean[c] = s / n;
    }
    Mat cov_dist(d, d);
    for (int j = 0; j < d; ++j)
        for (int i = 0; i < d; ++i) {
            double s = 0;
            for (int r : rows) s += (feat(r, i) - colmean[i]) * (feat(r, j) - colmean[j]);
            cov_dist(i, j) = s / (n - 1);
        }

    Mat avg(d, d);
    for (size_t i = 0; i < avg.numel(); ++i) avg.d[i] = (model.cov.d[i] + cov_dist.d[i]) / 2;
    const Mat inv = mc::pinv_sym(avg);
    Vec diff(d);
    for (int i = 0; i < d; ++i) diff[i] = model.mu[i] - mu_dist[i];
    double q = 0;
    for (int j = 0; j < d; ++j) {
        double t = 0;
        for (int i = 0; i < d; ++i) t += diff[i] * inv(i, j);
        q += t * diff[j];
    }
    return std::sqrt(q);
}
