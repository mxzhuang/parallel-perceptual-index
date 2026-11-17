#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
// ============================================================================
// M2: block-DCT statistics (block_dct.m and the per-block functions)
// ============================================================================
double std_over_mean(const double* a, size_t n, double eps) {
    return mc::stdev(a, n) / (mc::mean(a, n) + eps);
}

// mean of sorted values in [from, to)
double mean_range(const Vec& s, size_t from, size_t to) { return mc::mean(s.data() + from, to - from); }

PI_NOINLINE void block_dct(const Mat& im, double out[6], const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    const int R = im.rows, C = im.cols;
    const int nbr = (R + 2) / 3, nbc = (C + 2) / 3;
    const size_t nb = static_cast<size_t>(nbr) * nbc;
    Vec gama(nb), cv(nb), ori(nb);
    static const mc::Dct2 dct(7);
    const GgdTable& ggd = ggd_table();

    // index lists of oriented1/2/3_dct_rho_config3 for a 7x7 block (0-based row, col)
    static const int o1[][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {1, 2}, {1, 3},
                                {1, 4}, {1, 5}, {1, 6}, {2, 4}, {2, 5}, {2, 6}, {3, 5}, {3, 6}};
    static const int o2[][2] = {{1, 1}, {2, 2}, {2, 3}, {3, 2}, {3, 3}, {3, 4}, {4, 3}, {4, 4},
                                {4, 5}, {4, 6}, {5, 4}, {5, 5}, {5, 6}, {6, 4}, {6, 5}, {6, 6}};
    static const int o3[][2] = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {2, 1}, {3, 1},
                                {4, 1}, {5, 1}, {6, 1}, {4, 2}, {5, 2}, {6, 2}, {5, 3}, {6, 3}};

    double win[49], D[49], a48[48], a16[16];
    for (int bc = 0; bc < nbc; ++bc)
        for (int br = 0; br < nbr; ++br) {
            // blkproc(im, [3 3], [2 2]): 7x7 window, zero outside the image
            for (int c = 0; c < 7; ++c)
                for (int r = 0; r < 7; ++r) {
                    int y = 3 * br - 2 + r, x = 3 * bc - 2 + c;
                    win[c * 7 + r] = (y >= 0 && y < R && x >= 0 && x < C) ? im(y, x) : 0.0;
                }
            // baseline: one DCT shared by all five statistics; faithful: one DCT per statistic,
            // as each of the five blkproc calls of block_dct.m recomputes dct2
            dct.apply(win, D);
            const size_t k = br + static_cast<size_t>(bc) * nbr;

            // all coefficients except DC (column-major)
            gama[k] = faithful ? GgdTable::fit_faithful(D + 1, 48) : ggd.fit(D + 1, 48);
            if (faithful) dct.apply(win, D);
            for (int i = 0; i < 48; ++i) a48[i] = std::fabs(D[i + 1]);
            cv[k] = std_over_mean(a48, 48, 0.0000001);

            double g[3];
            const int (*lists[3])[2] = {o1, o2, o3};
            for (int l = 0; l < 3; ++l) {
                if (faithful) dct.apply(win, D);
                for (int i = 0; i < 16; ++i) a16[i] = std::fabs(D[lists[l][i][1] * 7 + lists[l][i][0]]);
                g[l] = std_over_mean(a16, 16, 0.00000001);
            }
            ori[k] = mc::var(g, 3);
        }

    std::sort(gama.begin(), gama.end());
    std::sort(cv.begin(), cv.end());
    std::sort(ori.begin(), ori.end());
    const double cnt = static_cast<double>(nb);
    const size_t first10 = static_cast<size_t>(std::ceil(cnt * 0.1));
    const size_t last10 = static_cast<size_t>(std::trunc(cnt * 0.9));  // MATLAB fix(), 1-based start
    out[0] = mean_range(gama, 0, first10);
    out[1] = mean_range(gama, 0, nb);
    out[2] = mean_range(cv, last10 - 1, nb);
    out[3] = mean_range(cv, 0, nb);
    out[4] = mean_range(ori, last10 - 1, nb);
    out[5] = mean_range(ori, 0, nb);
}
}  // namespace

void m2_block_dct(const SpatialPyramid& sp, double f1[18], const Config& cfg) {
    for (int s = 0; s < 3; ++s) block_dct(sp.im[s], &f1[6 * s], cfg);
}
