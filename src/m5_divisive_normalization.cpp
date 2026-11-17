#include "modules.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
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

// Inverse of the second-moment matrix after clipping its negative eigenvalues (the trace is
// kept). At most 15 x 15, so it stays serial in every version.
Mat precision_matrix(const Mat& Cx) {
    const int ncols = Cx.rows;
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
    return mc::pinv_sym(Cc);
}

// ---- designed parallel version -------------------------------------------------------
constexpr int kMaxCols = 15;                           // 9 neighbours, parent, 5 orientations
constexpr int kTri = kMaxCols * (kMaxCols + 1) / 2;    // upper triangle of the second moments
constexpr int kCentre = 4;                             // entry (ny, nx) = (0, 0) of the 3x3 block

// Where the entries of a neighbourhood vector come from: entry n of the vector at interior
// position (r, c) is element (r + row offset, c + column offset) of source matrix n. The
// sources are the band itself (nine shifts), the upsampled parent and the other five
// orientations; all of them are only read while a band is processed.
struct Neighbourhood {
    int ncols = 0;
    int nblv = 0, nblh = 0;            // interior positions per image column / image columns
    const double* base[kMaxCols] = {};  // element of source n used by position (0, 0)
    long stride[kMaxCols] = {};         // rows of source n = distance between image columns

    void add(const Mat& m, int roff, int coff) {
        base[ncols] = m.d.data() + static_cast<size_t>(coff) * m.rows + roff;
        stride[ncols] = m.rows;
        ++ncols;
    }
    long positions() const { return static_cast<long>(nblv) * nblh; }
    double centre(long k) const {
        const long c = k / nblv;
        return base[kCentre][c * stride[kCentre] + (k - c * nblv)];
    }
};

// Row source that reads the neighbourhood vector straight from the bands. Nothing is stored,
// so the working set of a pass is the band, its parent and five neighbours (seven images)
// instead of fifteen images' worth of copies.
struct FusedRows {
    const Neighbourhood& nb;
    void operator()(long, long r, long c, double* y) const {
        for (int n = 0; n < nb.ncols; ++n) y[n] = nb.base[n][c * nb.stride[n] + r];
    }
};

// Row source that reads row k of the stored neighbourhood matrix Y (positions x ncols,
// column-major): the same values as FusedRows, fetched from ncols separate streams.
struct StoredRows {
    const double* Y;
    long nexp;
    int ncols;
    void operator()(long k, long, long, double* y) const {
        for (int n = 0; n < ncols; ++n) y[n] = Y[n * nexp + k];
    }
};

// Y for the materialized variant. The loop unit is one image column of one column of Y, so
// every chunk fills one contiguous block of Y and the blocks of different threads meet at
// most at one cache line. The storage is not zero-filled first: each element is written
// exactly once here, and that first write also places the page near the thread that fills it.
std::unique_ptr<double[]> materialize(const Neighbourhood& nb, const Config& cfg) {
    const long nexp = nb.positions();
    std::unique_ptr<double[]> Y(new double[static_cast<size_t>(nb.ncols) * nexp]);
    par_for(cfg, static_cast<long>(nb.ncols) * nb.nblh, Sched{}, [&](long lo, long hi) {
        for (long u = lo; u < hi; ++u) {
            const long n = u / nb.nblh, c = u % nb.nblh;
            std::copy_n(nb.base[n] + c * nb.stride[n], nb.nblv, Y.get() + n * nexp + c * nb.nblv);
        }
    });
    return Y;
}

// C_x = Y'Y / nexp, with the positions (rows of Y) divided among the threads.
//
// Every entry of C_x is a sum over all positions. Giving each thread some of the 120 entries
// would make each of them sweep two full columns of Y per entry, 240 passes over data that
// does not fit in any cache. Here the data is read once: a chunk of positions updates all
// 120 partial sums, which stay in L1.
//
// Race-free without locks or atomics: a chunk accumulates into locals and stores them once
// into its own cache-line aligned slot. The chunk grid depends on nexp only and the slots
// are added in chunk order, so the sums are the same for any thread count and schedule, and
// for both row sources. Iterations are independent; only the additions inside one chunk
// form a chain, which is why each chunk walks its positions in index order.
template <class Rows>
Mat second_moments(const Config& cfg, const Neighbourhood& nb, const Rows& rows) {
    const long nexp = nb.positions();
    const int ncols = nb.ncols, nblv = nb.nblv;
    const long nc = reduce_chunks(nexp);
    std::vector<Padded<std::array<double, kTri>>> part(nc);
    par_for(cfg, nc, Sched{}, [&](long clo, long chi) {
        for (long ch = clo; ch < chi; ++ch) {
            long lo, hi;
            chunk_bounds(nexp, nc, ch, lo, hi);
            std::array<double, kTri> acc{};
            double y[kMaxCols];
            long c = lo / nblv, r = lo % nblv;
            for (long k = lo; k < hi; ++k) {
                rows(k, r, c, y);
                double* a = acc.data();  // column j of the packed upper triangle
                for (int j = 0; j < ncols; ++j) {
                    const double yj = y[j];
                    for (int i = 0; i <= j; ++i) a[i] += y[i] * yj;
                    a += j + 1;
                }
                if (++r == nblv) {
                    r = 0;
                    ++c;
                }
            }
            part[ch].v = acc;
        }
    });
    Mat Cx(ncols, ncols);
    for (int j = 0, t = 0; j < ncols; ++j)
        for (int i = 0; i <= j; ++i, ++t) {
            double s = 0;
            for (long ch = 0; ch < nc; ++ch) s += part[ch].v[t];
            Cx(i, j) = Cx(j, i) = s / static_cast<double>(nexp);
        }
    return Cx;
}

// g = (centre - mu) / sqrt(y' P y / N) for every position, written directly into the output
// without its guard band of gb positions on each side. Positions are independent and cost
// the same, so they are split into one contiguous block per thread (static schedule). The
// output index grows with the position index, so each block writes its own contiguous range
// of sb; P and the sources are read-only. Guard-band positions are still evaluated, because
// a zero energy anywhere in the band is an error.
template <class Rows>
void normalize(const Config& cfg, const Neighbourhood& nb, const Rows& rows, const Mat& P, int N, double mu, int gb,
               int rr, int cc, double* sb) {
    const int ncols = nb.ncols, nblv = nb.nblv;
    const double* Pd = P.d.data();
    par_for(cfg, nb.positions(), Sched{}, [&](long lo, long hi) {
        double y[kMaxCols];
        long c = lo / nblv, r = lo % nblv;
        for (long k = lo; k < hi; ++k) {
            rows(k, r, c, y);
            double s = 0;
            for (int j = 0; j < ncols; ++j) {
                const double* pj = Pd + j * ncols;
                double t = 0;
                for (int i = 0; i < ncols; ++i) t += y[i] * pj[i];
                s += t * y[j] / N;
            }
            const double z = std::sqrt(s);
            if (z == 0) throw std::runtime_error("zero local energy in divisive normalisation");
            const long ro = r - gb, co = c - gb;
            if (ro >= 0 && ro < rr && co >= 0 && co < cc) sb[ro + co * rr] = (y[kCentre] - mu) / z;
            if (++r == nblv) {
                r = 0;
                ++c;
            }
        }
    });
}

// One band of the designed version. Writes only sb and size.
void normalize_band(const SteerablePyramid& pyr, int scale, int orien, const Config& cfg, Vec& sb, int size[2]) {
    const int Nor = 6, Nband = 13;
    const int nband = scale * Nor + orien + 2;
    const Mat& aux = pyr.band[scale][orien];
    const bool prnt = nband < Nband - Nor;
    Mat auxp;
    if (prnt) auxp = mc::imresize_scale(pyr.band[scale + 1][orien], 2.0, cfg);

    Neighbourhood nb;
    nb.nblv = aux.rows - 2;
    nb.nblh = aux.cols - 2;
    for (int ny = -1; ny <= 1; ++ny)
        for (int nx = -1; nx <= 1; ++nx) nb.add(aux, 1 - ny, 1 - nx);
    if (prnt) nb.add(auxp, 1, 1);
    for (int neib = 0; neib < Nor; ++neib)
        if (neib != orien) nb.add(pyr.band[scale][neib], 1, 1);
    const long nexp = nb.positions();
    const int N = 9 + (prnt ? 1 : 0);

    const int gb = 16 >> scale;
    const int rr = nb.nblv - 2 * gb, cc = nb.nblh - 2 * gb;
    sb.resize(static_cast<size_t>(rr) * cc);
    const double mu = det_sum(cfg, nexp, [&](long k) { return nb.centre(k); }) / static_cast<double>(nexp);

    // Both variants run the same two passes on the same chunk grid and differ only in where
    // a neighbourhood vector is read from, so their results agree bit for bit.
    if (cfg.m5_gather == M5Gather::Materialized) {
        const std::unique_ptr<double[]> Y = materialize(nb, cfg);
        const StoredRows rows{Y.get(), nexp, nb.ncols};
        const Mat P = precision_matrix(second_moments(cfg, nb, rows));
        normalize(cfg, nb, rows, P, N, mu, gb, rr, cc, sb.data());
    } else {
        const FusedRows rows{nb};
        const Mat P = precision_matrix(second_moments(cfg, nb, rows));
        normalize(cfg, nb, rows, P, N, mu, gb, rr, cc, sb.data());
    }

    const long nsb = static_cast<long>(sb.size());
    const double m2 = det_sum(cfg, nsb, [&](long i) { return sb[i]; }) / static_cast<double>(nsb);
    par_for(cfg, nsb, Sched{}, [&](long lo, long hi) {
        for (long i = lo; i < hi; ++i) sb[i] -= m2;
    });
    size[0] = rr;
    size[1] = cc;
}
}  // namespace

PI_NOINLINE NormalizedBands m5_divisive_normalization(const SteerablePyramid& pyr, const Config& cfg) {
    NormalizedBands out;
    const int Nor = 6, Nband = 13;  // size(pind,1) - 1
    if (cfg.designed()) {
        // The 12 bands are independent and each one writes only its own output slot. As tasks
        // they overlap inside the task graph; otherwise they run in turn and the position
        // loops inside a band get every thread.
        par_outer(cfg, 2 * Nor, [&](int band) {
            normalize_band(pyr, band / Nor, band % Nor, cfg, out.subband[band], out.size_band[band]);
        });
        return out;
    }
    // Sequential baseline; the naive version is the same code with `parallel for` on its two
    // hot loops.
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
            // naive: each j writes only its own entries (i, j) and (j, i), i <= j
            #pragma omp parallel for num_threads(cfg.threads) if(cfg.naive())
            for (int j = 0; j < ncols; ++j)
                for (int i = 0; i <= j; ++i) {
                    double s = 0;
                    const double* yi = &Y.d[static_cast<size_t>(i) * nexp];
                    const double* yj = &Y.d[static_cast<size_t>(j) * nexp];
                    for (size_t k = 0; k < nexp; ++k) s += yi[k] * yj[k];
                    Cx(i, j) = Cx(j, i) = s / static_cast<double>(nexp);
                }
            const Mat P = precision_matrix(Cx);

            // z = sqrt(sum((Y * P) .* Y / N, 2))
            Vec o_c(nexp);
            for (int c = 0; c < nblh; ++c)
                for (int r = 0; r < nblv; ++r) o_c[r + static_cast<size_t>(c) * nblv] = aux(r + 1, c + 1);
            const double mu = mc::mean(o_c.data(), nexp);
            for (double& v : o_c) v -= mu;
            if (cfg.variant == Variant::Faithful) unused_histogram(o_c, 50);
            Vec g(nexp);
            std::vector<double> yrow(ncols);
            // naive: every thread gets its own copy of the scratch row; an exception must not
            // leave the parallel region, so it is kept and rethrown after the loop
            detail::ErrorSlot err;
            #pragma omp parallel for num_threads(cfg.threads) if(cfg.naive()) firstprivate(yrow)
            for (size_t k = 0; k < nexp; ++k) {
                try {
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
                } catch (...) {
                    err.capture();
                }
            }
            err.rethrow();
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
