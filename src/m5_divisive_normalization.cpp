#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
// ============================================================================
// M5: divisive normalisation (norm_sender_normalized.m, Nsc = 2, Nor = 6, 3x3 block,
//     parent and neighbours on). The histogram of the original is never used; it is computed
//     only in the faithful variant.
// ============================================================================
// [hy, rangeo] = hist(o_c, 50); hy = hy / sum(hy)  (result unused by the official code)
void unused_histogram(const Vec& x, int nbins) {
    double lo = x[0], hi = x[0];
    for (double v : x) {
        lo = std::min(lo, v);
        hi = std::max(hi, v);
    }
    std::vector<double> hy(nbins, 0.0);
    const double w = (hi - lo) / nbins;
    for (double v : x) {
        int b = w > 0 ? static_cast<int>((v - lo) / w) : 0;
        hy[std::min(std::max(b, 0), nbins - 1)] += 1;
    }
    double total = 0;
    for (double h : hy) total += h;
    for (double& h : hy) h /= total;
    g_unused_sink = hy[0];
}
}  // namespace

PI_NOINLINE NormalizedBands m5_divisive_normalization(const SteerablePyramid& pyr, const Config& cfg) {
    NormalizedBands out;
    const int Nor = 6, Nband = 13;  // size(pind,1) - 1
    int p = 0;
    for (int scale = 0; scale < 2; ++scale)
        for (int orien = 0; orien < Nor; ++orien, ++p) {
            const int nband = scale * Nor + orien + 2;  // 1-based pyramid band index
            const Mat& aux = pyr.band[scale][orien];
            const int Nsy = aux.rows, Nsx = aux.cols;
            const bool prnt = nband < Nband - Nor;  // as in the original: only nband 2..6
            Mat auxp;
            if (prnt) auxp = mc::imresize_scale(pyr.band[scale + 1][orien], 2.0);  // cropped on use

            const int nblv = Nsy - 2, nblh = Nsx - 2;
            const size_t nexp = static_cast<size_t>(nblv) * nblh;
            const int N = 9 + (prnt ? 1 : 0);  // divisor used by the original
            const int ncols = N + (Nor - 1);
            Mat Y(static_cast<int>(nexp), ncols);
            int n = 0;
            for (int ny = -1; ny <= 1; ++ny)
                for (int nx = -1; nx <= 1; ++nx, ++n)
                    for (int c = 0; c < nblh; ++c)
                        for (int r = 0; r < nblv; ++r) Y(r + c * nblv, n) = aux(r + 1 - ny, c + 1 - nx);
            if (prnt) {
                for (int c = 0; c < nblh; ++c)
                    for (int r = 0; r < nblv; ++r) Y(r + c * nblv, n) = auxp(r + 1, c + 1);
                ++n;
            }
            for (int neib = 0; neib < Nor; ++neib) {
                if (neib == orien) continue;
                const Mat& a1 = pyr.band[scale][neib];
                for (int c = 0; c < nblh; ++c)
                    for (int r = 0; r < nblv; ++r) Y(r + c * nblv, n) = a1(r + 1, c + 1);
                ++n;
            }

            // C_x = Y'Y / nexp
            Mat Cx(ncols, ncols);
            for (int j = 0; j < ncols; ++j)
                for (int i = 0; i <= j; ++i) {
                    double s = 0;
                    const double* yi = &Y.d[static_cast<size_t>(i) * nexp];
                    const double* yj = &Y.d[static_cast<size_t>(j) * nexp];
                    for (size_t k = 0; k < nexp; ++k) s += yi[k] * yj[k];
                    Cx(i, j) = Cx(j, i) = s / static_cast<double>(nexp);
                }
            // clip negative eigenvalues, keep the trace, rebuild
            Vec lam;
            Mat Q;
            mc::eig_sym(Cx, lam, Q);
            double total = 0, pos = 0;
            for (double l : lam) {
                total += l;
                pos += l * (l > 0);
            }
            const double denom = pos + (pos == 0 ? 1.0 : 0.0);
            for (double& l : lam) l = (l * (l > 0)) * total / denom;
            Mat Cc(ncols, ncols);
            for (int j = 0; j < ncols; ++j)
                for (int i = 0; i < ncols; ++i) {
                    double s = 0;
                    for (int k = 0; k < ncols; ++k) s += Q(i, k) * lam[k] * Q(j, k);
                    Cc(i, j) = s;
                }
            const Mat P = mc::pinv_sym(Cc);

            // z = sqrt(sum((Y * P) .* Y / N, 2))
            Vec o_c(nexp);
            for (int c = 0; c < nblh; ++c)
                for (int r = 0; r < nblv; ++r) o_c[r + static_cast<size_t>(c) * nblv] = aux(r + 1, c + 1);
            const double mu = mc::mean(o_c.data(), nexp);
            for (double& v : o_c) v -= mu;
            if (cfg.variant == Variant::Faithful) unused_histogram(o_c, 50);
            Vec g(nexp);
            std::vector<double> yrow(ncols), yp(ncols);
            for (size_t k = 0; k < nexp; ++k) {
                for (int j = 0; j < ncols; ++j) yrow[j] = Y.d[static_cast<size_t>(j) * nexp + k];
                double s = 0;
                for (int j = 0; j < ncols; ++j) {
                    double t = 0;
                    for (int i = 0; i < ncols; ++i) t += yrow[i] * P(i, j);
                    s += t * yrow[j] / N;
                }
                const double z = std::sqrt(s);
                if (z == 0) throw std::runtime_error("zero local energy in divisive normalisation");
                g[k] = o_c[k] / z;
            }
            // remove the guard band and the mean
            const int gb = 16 >> scale;
            const int rr = nblv - 2 * gb, cc = nblh - 2 * gb;
            Vec& sb = out.subband[p];
            sb.resize(static_cast<size_t>(rr) * cc);
            for (int c = 0; c < cc; ++c)
                for (int r = 0; r < rr; ++r) sb[r + static_cast<size_t>(c) * rr] = g[(r + gb) + static_cast<size_t>(c + gb) * nblv];
            const double m2 = mc::mean(sb.data(), sb.size());
            for (double& v : sb) v -= m2;
            out.size_band[p][0] = rr;
            out.size_band[p][1] = cc;
        }
    return out;
}
