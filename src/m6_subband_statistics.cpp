#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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

// mcs = mean2(cs_map), cs_map = (2 sigma12 + C2) / (sigma1^2 + sigma2^2 + C2)
// Baseline: separable window, only the structure term. Faithful: direct 11x11 window and the
// full ssim_map / mssim as well, which the official code computes but Ma does not use.
PI_NOINLINE double ssim_structure(const Mat& a, const Mat& b, const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    const SsimWindow& w = ssim_window();
    auto filt = [&](const Mat& x) {
        return faithful ? mc::filter2_valid(w.w2d, x) : mc::filter2_valid_separable(w.w1d, x);
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
}  // namespace

PI_NOINLINE void m6a_band_highpass_ssim(const SteerablePyramid& pyr, double out[12], const Config& cfg) {
    for (int i = 0; i < 12; ++i) {
        const Mat& band = pyr.band[i / 6][i % 6];
        Mat up = mc::imresize_size(band, pyr.hi0.rows, pyr.hi0.cols);
        out[i] = ssim_structure(up, pyr.hi0, cfg);
    }
}

PI_NOINLINE void m6b_normalized_band_stats(const NormalizedBands& nb, double ggd_out[18], double ssim_out[15],
                                           const Config& cfg) {
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
