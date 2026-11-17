#include "modules.h"

#include <algorithm>
#include <cmath>

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

// Statistics of the windows in columns [bc_lo, bc_hi). The 7x7 window, its DCT and the
// scratch arrays are locals of the call, so each chunk of a parallel loop has its own
// copies; the image and the tables are only read. Outputs are indexed k = br + bc * nbr:
// a range of window columns is one contiguous slice of gama, cv and ori that no other
// chunk writes.
void window_columns(const Mat& im, int bc_lo, int bc_hi, bool faithful, Vec& gama, Vec& cv, Vec& ori) {
    const int R = im.rows, C = im.cols;
    const int nbr = (R + 2) / 3;
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
    for (int bc = bc_lo; bc < bc_hi; ++bc)
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
}

PI_NOINLINE void block_dct(const Mat& im, double out[6], const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    const int nbr = (im.rows + 2) / 3, nbc = (im.cols + 2) / 3;
    const size_t nb = static_cast<size_t>(nbr) * nbc;
    Vec gama(nb), cv(nb), ori(nb);
    Vec* stat[3] = {&gama, &cv, &ori};

    if (cfg.designed()) {
        // Windows are independent and cost about the same, so a static block partition
        // balances the columns with one chunk per thread and a single false-sharing border
        // between neighbours. schedule(static,1) deals the columns out cyclically instead:
        // neighbouring columns go to different threads, every slice border becomes a cache
        // line written by two threads, the image columns shared by overlapping windows are
        // loaded into several caches, and each column pays the cost of starting a chunk.
        par_for(cfg, nbc, cfg.m2_sched, [&](long lo, long hi) {
            window_columns(im, static_cast<int>(lo), static_cast<int>(hi), faithful, gama, cv, ori);
        }, cfg.load_m2);
        // The 10% tails are defined over the whole scale with exact sample counts, so the
        // chunks cannot pool their own slices. The three full sorts are independent (one
        // array each) and of unequal cost: three units handed out one at a time.
        par_for(cfg, 3, Sched{SchedKind::Dynamic, 1}, [&](long lo, long hi) {
            for (long i = lo; i < hi; ++i) std::sort(stat[i]->begin(), stat[i]->end());
        });
    } else {
        window_columns(im, 0, nbc, faithful, gama, cv, ori);
        for (Vec* v : stat) std::sort(v->begin(), v->end());
    }

    // sums over sorted values, in index order: the same for every thread count
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
    // Each scale writes its own six outputs. In the task graph the three scales overlap;
    // elsewhere they run in turn and the window loop gets every thread.
    par_outer(cfg, 3, [&](int s) { block_dct(sp.im[s], &f1[6 * s], cfg); });
}
