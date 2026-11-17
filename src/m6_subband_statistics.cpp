#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
// ============================================================================
// M6: subband statistics (rest of global_gsm.m) with ssim_index_new's structure term
// ============================================================================
struct SsimWindow {
    Mat w2d;  // fspecial('gaussian', 11, 1.5) / sum(sum(.))
    Vec w1d;  // separable factor
};

const SsimWindow& ssim_window() {
    static const SsimWindow w = [] {
        SsimWindow s;
        s.w2d = mc::fspecial_gaussian(11, 1.5);
        double total = 0;
        for (int c = 0; c < 11; ++c) {
            double cs = 0;
            for (int r = 0; r < 11; ++r) cs += s.w2d(r, c);
            total += cs;
        }
        for (double& v : s.w2d.d) v /= total;
        s.w1d = mc::gaussian_1d(11, 1.5);
        return s;
    }();
    return w;
}

// mc::filter2_valid_separable with a `parallel for` on each pass, for the naive version.
// Every iteration writes one column of its output and reads shared data only.
Mat filter2_valid_separable_naive(const Vec& k, const Mat& a, const Config& cfg) {
    (void)cfg;  // only named in the pragmas, which a build without OpenMP ignores
    const int n = static_cast<int>(k.size());
    const int R = a.rows - n + 1, C = a.cols - n + 1;
    if (R <= 0 || C <= 0) return Mat();
    Mat t(R, a.cols);  // vertical pass
    #pragma omp parallel for num_threads(cfg.threads)
    for (int c = 0; c < a.cols; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += k[i] * a(r + i, c);
            t(r, c) = s;
        }
    Mat o(R, C);  // horizontal pass
    #pragma omp parallel for num_threads(cfg.threads)
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int j = 0; j < n; ++j) s += k[j] * t(r, c + j);
            o(r, c) = s;
        }
    return o;
}

// mcs = mean2(cs_map), cs_map = (2 sigma12 + C2) / (sigma1^2 + sigma2^2 + C2)
// Baseline: separable window, only the structure term. Faithful: direct 11x11 window and the
// full ssim_map / mssim as well, which the official code computes but Ma does not use.
// Naive: the baseline with `parallel for` on the filter passes and on the per-pixel loop.
PI_NOINLINE double ssim_structure(const Mat& a, const Mat& b, const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    const SsimWindow& w = ssim_window();
    auto filt = [&](const Mat& x) {
        if (faithful) return mc::filter2_valid(w.w2d, x);
        return cfg.naive() ? filter2_valid_separable_naive(w.w1d, x, cfg) : mc::filter2_valid_separable(w.w1d, x);
    };
    Mat aa(a.rows, a.cols), bb(a.rows, a.cols), ab(a.rows, a.cols);
    for (size_t i = 0; i < a.numel(); ++i) {
        aa.d[i] = a.d[i] * a.d[i];
        bb.d[i] = b.d[i] * b.d[i];
        ab.d[i] = a.d[i] * b.d[i];
    }
    const Mat mu1 = filt(a), mu2 = filt(b), s11 = filt(aa), s22 = filt(bb), s12 = filt(ab);
    const double C1 = (0.01 * 255) * (0.01 * 255);
    const double C2 = (0.03 * 255) * (0.03 * 255);
    double sum = 0, ssim_sum = 0;
    // naive only: the sums are the one shared update, so they become reductions
    #pragma omp parallel for num_threads(cfg.threads) reduction(+ : sum, ssim_sum) if (cfg.naive())
    for (size_t i = 0; i < mu1.numel(); ++i) {
        double m1 = mu1.d[i], m2 = mu2.d[i];
        double sigma1_sq = s11.d[i] - m1 * m1;
        double sigma2_sq = s22.d[i] - m2 * m2;
        double sigma12 = s12.d[i] - m1 * m2;
        sum += (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
        if (faithful)
            ssim_sum += ((2 * m1 * m2 + C1) * (2 * sigma12 + C2)) / ((m1 * m1 + m2 * m2 + C1) * (sigma1_sq + sigma2_sq + C2));
    }
    if (faithful) g_unused_sink = ssim_sum / static_cast<double>(mu1.numel());  // mssim
    return sum / static_cast<double>(mu1.numel());
}

Mat reshape(const Vec& v, int rows, int cols) {
    Mat m(rows, cols);
    m.d = v;
    return m;
}

// ============================================================================
// Designed parallel versions
// ============================================================================
// The mean of the structure map is always taken on one grid of column strips that depends
// on the image width only: each strip is summed in column-major order and the strip sums
// are added in strip order. The three strategies below share this grid and the per-column
// kernels, so they return the same bits for any thread count and schedule. Each pixel's
// value is that of the baseline; only the grouping of the final sum differs from it.

constexpr int kTaps = 11;        // length of the SSIM window
constexpr int kMaps = 5;         // filtered maps: a, b, a*a, b*b, a*b
constexpr int kStripCols = 64;   // output columns per strip: 10 halo columns cost 16% extra

// A column-major image that is not owned (a matrix, or a vectorised band with its size).
struct Image {
    const double* d;
    int rows, cols;
    const double* col(int c) const { return d + static_cast<size_t>(c) * rows; }
};

Image view(const Mat& m) { return {m.d.data(), m.rows, m.cols}; }
Image view(const NormalizedBands& nb, int i) { return {nb.subband[i].data(), nb.size_band[i][0], nb.size_band[i][1]}; }

struct SsimPair {
    Image a, b;
    int out_rows() const { return std::max(0, a.rows - kTaps + 1); }
    int out_cols() const { return std::max(0, a.cols - kTaps + 1); }
    long strips() const { return out_rows() == 0 ? 0 : (out_cols() + kStripCols - 1) / kStripCols; }
    void strip_bounds(long s, int& c0, int& c1) const {
        c0 = static_cast<int>(s) * kStripCols;
        c1 = std::min(out_cols(), c0 + kStripCols);
    }
    double mean(double total) const { return total / (static_cast<double>(out_rows()) * out_cols()); }
};

// Vertical pass of the five maps for one input column: out holds kMaps blocks of R values.
// Taps are added in ascending order for each element, as in mc::filter2_valid_separable.
// prod is scratch for the three product columns (3 * rows).
inline void vertical_pass(const double* k, const double* a, const double* b, int rows, int R, double* prod,
                          double* out) {
    double* aa = prod;
    double* bb = prod + rows;
    double* ab = prod + 2 * static_cast<size_t>(rows);
    for (int r = 0; r < rows; ++r) {
        aa[r] = a[r] * a[r];
        bb[r] = b[r] * b[r];
        ab[r] = a[r] * b[r];
    }
    const double* in[kMaps] = {a, b, aa, bb, ab};
    for (int m = 0; m < kMaps; ++m) {
        const double* x = in[m];
        double* t = out + static_cast<size_t>(m) * R;
        std::fill(t, t + R, 0.0);
        for (int i = 0; i < kTaps; ++i)
            for (int r = 0; r < R; ++r) t[r] += k[i] * x[r + i];
    }
}

// Horizontal pass for one output column: src[j] is the vertical pass of input column c + j.
// All five maps are handled as one vector of n = kMaps * R elements.
inline void horizontal_pass(const double* k, const double* const src[kTaps], size_t n, double* out) {
    std::fill(out, out + n, 0.0);
    for (int j = 0; j < kTaps; ++j) {
        const double* t = src[j];
        for (size_t i = 0; i < n; ++i) out[i] += k[j] * t[i];
    }
}

// Adds the structure terms of one output column (five filtered maps of R values) to sum.
inline double structure_column(const double* f, int R, double sum) {
    const double C2 = (0.03 * 255) * (0.03 * 255);
    const double* mu1 = f;
    const double* mu2 = f + R;
    const double* s11 = f + 2 * static_cast<size_t>(R);
    const double* s22 = f + 3 * static_cast<size_t>(R);
    const double* s12 = f + 4 * static_cast<size_t>(R);
    for (int r = 0; r < R; ++r) {
        double m1 = mu1[r], m2 = mu2[r];
        double sigma1_sq = s11[r] - m1 * m1;
        double sigma2_sq = s22[r] - m2 * m2;
        double sigma12 = s12[r] - m1 * m2;
        sum += (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
    }
    return sum;
}

// Buffers of one strip. The vertical pass of the last 11 input columns is kept in a ring,
// so a strip touches 11 + 1 columns of the five maps (about 0.5 MB for 1080 rows) however
// wide it is, instead of five full-size maps.
struct StripScratch {
    Vec prod, ring, out;
    void fit(const SsimPair& p) {
        const size_t block = static_cast<size_t>(kMaps) * p.out_rows();
        if (prod.size() < 3 * static_cast<size_t>(p.a.rows)) prod.resize(3 * static_cast<size_t>(p.a.rows));
        if (ring.size() < kTaps * block) ring.resize(kTaps * block);
        if (out.size() < block) out.resize(block);
    }
};

// Sum of the structure map over output columns [c0, c1). Reads input columns
// [c0, c1 + 10) of the shared images and writes only the scratch of the caller.
double strip_sum(const SsimPair& p, int c0, int c1, StripScratch& s) {
    const double* k = ssim_window().w1d.data();
    const int R = p.out_rows();
    const size_t block = static_cast<size_t>(kMaps) * R;
    s.fit(p);
    double sum = 0;
    for (int c = c0; c < c1 + kTaps - 1; ++c) {
        vertical_pass(k, p.a.col(c), p.b.col(c), p.a.rows, R, s.prod.data(), &s.ring[(c - c0) % kTaps * block]);
        if (c - c0 < kTaps - 1) continue;  // halo: the window of the first output column is not full yet
        const int first = c - c0 - (kTaps - 1);
        const double* src[kTaps];
        for (int j = 0; j < kTaps; ++j) src[j] = &s.ring[(first + j) % kTaps * block];
        horizontal_pass(k, src, block, s.out.data());
        sum = structure_column(s.out.data(), R, sum);
    }
    return sum;
}

// Tasks strategy: one whole SSIM, strips one after another on the calling thread.
double ssim_serial(const SsimPair& p) {
    StripScratch scratch;
    double total = 0;
    for (long s = 0; s < p.strips(); ++s) {
        int c0, c1;
        p.strip_bounds(s, c0, c1);
        total += strip_sum(p, c0, c1, scratch);
    }
    return p.mean(total);
}

// Inner strategy: full-size maps, each pass a parallel loop over columns. Columns are
// independent in every pass and a chunk writes only its own columns, so a static block
// partition into contiguous column ranges balances the uniform work without false sharing.
double ssim_inner(const SsimPair& p, const Config& cfg) {
    const double* k = ssim_window().w1d.data();
    const int R = p.out_rows(), C = p.out_cols();
    const size_t block = static_cast<size_t>(kMaps) * R;
    if (R == 0 || C == 0) return p.mean(0.0);
    Vec t(block * p.a.cols);
    par_for(cfg, p.a.cols, Sched{}, [&](long lo, long hi) {
        Vec prod(3 * static_cast<size_t>(p.a.rows));  // once per chunk
        for (long c = lo; c < hi; ++c)
            vertical_pass(k, p.a.col(static_cast<int>(c)), p.b.col(static_cast<int>(c)), p.a.rows, R, prod.data(),
                          &t[c * block]);
    }, cfg.load_m6);
    Vec o(block * C);
    par_for(cfg, C, Sched{}, [&](long lo, long hi) {
        for (long c = lo; c < hi; ++c) {
            const double* src[kTaps];
            for (int j = 0; j < kTaps; ++j) src[j] = &t[(c + j) * block];
            horizontal_pass(k, src, block, &o[c * block]);
        }
    }, cfg.load_m6);
    std::vector<Padded<double>> part(p.strips());  // one cache line per strip sum
    par_for(cfg, p.strips(), Sched{}, [&](long lo, long hi) {
        for (long s = lo; s < hi; ++s) {
            int c0, c1;
            p.strip_bounds(s, c0, c1);
            double sum = 0;
            for (int c = c0; c < c1; ++c) sum = structure_column(&o[c * block], R, sum);
            part[s].v = sum;
        }
    }, cfg.load_m6);
    double total = 0;
    for (const Padded<double>& v : part) total += v.v;
    return p.mean(total);
}

// Strips strategy: the strips of all SSIMs form one flat list, so the team joins once and
// the work is cut into a few hundred units instead of n unequal ones. A strip keeps its sum
// in a local and stores it once in its own padded slot; the slots are merged after the loop
// in list order. Dynamic schedule, because the last strip of an image is narrower and the
// threads may be shared with other modules; the default chunk (about eight per thread)
// lets a chunk reuse one scratch allocation for all its strips.
void ssim_strips(const std::vector<SsimPair>& pairs, double* out, const Config& cfg) {
    std::vector<long> first(pairs.size() + 1, 0);
    for (size_t i = 0; i < pairs.size(); ++i) first[i + 1] = first[i] + pairs[i].strips();
    std::vector<Padded<double>> part(first.back());
    par_for(cfg, first.back(), Sched{SchedKind::Dynamic, 0}, [&](long lo, long hi) {
        StripScratch scratch;  // once per chunk
        size_t i = std::upper_bound(first.begin(), first.end(), lo) - first.begin() - 1;
        for (long u = lo; u < hi; ++u) {
            while (u >= first[i + 1]) ++i;
            int c0, c1;
            pairs[i].strip_bounds(u - first[i], c0, c1);
            part[u].v = strip_sum(pairs[i], c0, c1, scratch);
        }
    }, cfg.load_m6);
    for (size_t i = 0; i < pairs.size(); ++i) {
        double total = 0;
        for (long u = first[i]; u < first[i + 1]; ++u) total += part[u].v;
        out[i] = pairs[i].mean(total);
    }
}

// Whole SSIMs as units: 12 or 15 units of different cost, taken one at a time.
const Sched kUnitSched{SchedKind::Dynamic, 1};

void m6a_designed(const SteerablePyramid& pyr, double out[12], const Config& cfg) {
    const int rows = pyr.hi0.rows, cols = pyr.hi0.cols;
    const Image hi0 = view(pyr.hi0);
    Config serial = cfg;  // a resize inside a unit must not open loops of its own
    serial.threads = 1;
    switch (cfg.m6_ssim) {
        case M6Ssim::Tasks:
            // each unit writes one element of out, once
            par_for(cfg, 12, kUnitSched, [&](long lo, long hi) {
                for (long i = lo; i < hi; ++i) {
                    const Mat up = mc::imresize_size(pyr.band[i / 6][i % 6], rows, cols, serial);
                    out[i] = ssim_serial({view(up), hi0});
                }
            }, cfg.load_m6);
            break;
        case M6Ssim::Inner:
            par_outer(cfg, 12, [&](int i) {
                const Mat up = mc::imresize_size(pyr.band[i / 6][i % 6], rows, cols, cfg);
                out[i] = ssim_inner({view(up), hi0}, cfg);
            });
            break;
        case M6Ssim::Strips: {
            // The strips of a band need its resized image, so the 12 bands are resized first,
            // one band per unit, and stay alive for the strip loop: two joins and 12
            // full-size images, in place of the nine full-size maps of one baseline SSIM.
            std::vector<Mat> up(12);
            par_for(cfg, 12, kUnitSched, [&](long lo, long hi) {
                for (long i = lo; i < hi; ++i) up[i] = mc::imresize_size(pyr.band[i / 6][i % 6], rows, cols, serial);
            });
            std::vector<SsimPair> pairs;
            for (int i = 0; i < 12; ++i) pairs.push_back({view(up[i]), hi0});
            ssim_strips(pairs, out, cfg);
            break;
        }
    }
}

void m6b_designed(const NormalizedBands& nb, double ggd_out[18], double ssim_out[15], const Config& cfg) {
    const GgdTable& ggd = ggd_table();
    // 18 independent fits of unequal length. A fit sums its vector sequentially, exactly as
    // the baseline does: a fit is a table lookup, and a rounding difference in its moments
    // could select another entry. Each unit writes one element of ggd_out, once.
    par_for(cfg, 18, kUnitSched, [&](long lo, long hi) {
        for (long i = lo; i < hi; ++i) {
            if (i < 12) {
                ggd_out[i] = ggd.fit(nb.subband[i].data(), nb.subband[i].size());
            } else {
                const Vec& fine = nb.subband[i - 12];
                const Vec& coarse = nb.subband[i - 6];
                ggd_out[i] = ggd.fit_pair(fine.data(), fine.size(), coarse.data(), coarse.size());
            }
        }
    });
    std::vector<SsimPair> pairs;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j) pairs.push_back({view(nb, i), view(nb, j)});
    switch (cfg.m6_ssim) {
        case M6Ssim::Tasks:
            par_for(cfg, 15, kUnitSched, [&](long lo, long hi) {
                for (long i = lo; i < hi; ++i) ssim_out[i] = ssim_serial(pairs[i]);
            }, cfg.load_m6);
            break;
        case M6Ssim::Inner:
            par_outer(cfg, 15, [&](int i) { ssim_out[i] = ssim_inner(pairs[i], cfg); });
            break;
        case M6Ssim::Strips:
            ssim_strips(pairs, ssim_out, cfg);
            break;
    }
}
}  // namespace

PI_NOINLINE void m6a_band_highpass_ssim(const SteerablePyramid& pyr, double out[12], const Config& cfg) {
    if (cfg.designed()) {
        m6a_designed(pyr, out, cfg);
        return;
    }
    for (int i = 0; i < 12; ++i) {
        const Mat& band = pyr.band[i / 6][i % 6];
        Mat up = mc::imresize_size(band, pyr.hi0.rows, pyr.hi0.cols);
        out[i] = ssim_structure(up, pyr.hi0, cfg);
    }
}

PI_NOINLINE void m6b_normalized_band_stats(const NormalizedBands& nb, double ggd_out[18], double ssim_out[15],
                                           const Config& cfg) {
    if (cfg.designed()) {
        m6b_designed(nb, ggd_out, ssim_out, cfg);
        return;
    }
    const GgdTable& ggd = ggd_table();
    auto fit = [&](const Vec& x) {
        return cfg.variant == Variant::Faithful ? GgdTable::fit_faithful(x.data(), x.size()) : ggd.fit(x.data(), x.size());
    };
    for (int i = 0; i < 12; ++i) ggd_out[i] = fit(nb.subband[i]);
    for (int i = 0; i < 6; ++i) {
        Vec t2 = nb.subband[i];
        t2.insert(t2.end(), nb.subband[i + 6].begin(), nb.subband[i + 6].end());
        ggd_out[12 + i] = fit(t2);
    }
    int k = 0;
    for (int i = 0; i < 6; ++i)
        for (int j = i + 1; j < 6; ++j)
            ssim_out[k++] = ssim_structure(reshape(nb.subband[i], nb.size_band[i][0], nb.size_band[i][1]),
                                           reshape(nb.subband[j], nb.size_band[j][0], nb.size_band[j][1]), cfg);
}
