#include "fft.h"

#include <algorithm>
#include <cmath>

#include "parallel.h"

namespace mc {

namespace {
constexpr double kPi = 3.14159265358979323846;
}  // namespace

namespace {
bool is_pow2(size_t n) { return n && !(n & (n - 1)); }

// Precomputed tables for one transform length: radix-2 for powers of two, Bluestein
// (chirp-z through a power-of-two convolution) otherwise. Only the forward transform is
// implemented; the inverse uses ifft(x) = conj(fft(conj(x))) / n.
class FftPlan {
public:
    explicit FftPlan(size_t n) : n_(n), pow2_(is_pow2(n)) {
        size_t L = n;
        if (!pow2_) {
            m_ = 1;
            while (m_ < 2 * n - 1) m_ <<= 1;
            L = m_;
        }
        init_radix2(L);
        if (!pow2_) {
            chirp_.resize(n);
            for (size_t k = 0; k < n; ++k) {
                unsigned long long kk = (static_cast<unsigned long long>(k) * k) % (2ULL * n);
                chirp_[k] = std::polar(1.0, -kPi * static_cast<double>(kk) / static_cast<double>(n));
            }
            bhat_.assign(m_, cplx(0, 0));
            bhat_[0] = std::conj(chirp_[0]);
            for (size_t k = 1; k < n; ++k) bhat_[k] = bhat_[m_ - k] = std::conj(chirp_[k]);
            radix2(bhat_.data(), false);
        }
    }

    // in-place forward DFT of x[0..n), work must hold m() elements (unused for powers of two)
    void forward(cplx* x, cplx* work) const {
        if (n_ <= 1) return;
        if (pow2_) {
            radix2(x, false);
            return;
        }
        for (size_t k = 0; k < n_; ++k) work[k] = x[k] * chirp_[k];
        for (size_t k = n_; k < m_; ++k) work[k] = 0;
        radix2(work, false);
        for (size_t i = 0; i < m_; ++i) work[i] *= bhat_[i];
        radix2(work, true);
        const double s = 1.0 / static_cast<double>(m_);
        for (size_t k = 0; k < n_; ++k) x[k] = chirp_[k] * work[k] * s;
    }
    size_t work_size() const { return pow2_ ? 0 : m_; }

private:
    size_t n_, m_ = 0;
    bool pow2_;
    std::vector<size_t> rev_;
    std::vector<cplx> tw_;  // exp(-2 pi i k / L), k < L/2
    std::vector<cplx> chirp_, bhat_;

    void init_radix2(size_t L) {
        rev_.resize(L);
        for (size_t i = 0, j = 0; i < L; ++i) {
            rev_[i] = j;
            size_t bit = L >> 1;
            for (; bit && (j & bit); bit >>= 1) j ^= bit;
            j ^= bit;
        }
        tw_.resize(L / 2);
        for (size_t k = 0; k < L / 2; ++k) tw_[k] = std::polar(1.0, -2 * kPi * static_cast<double>(k) / static_cast<double>(L));
    }

    void radix2(cplx* a, bool inverse) const {
        const size_t L = rev_.size();
        for (size_t i = 0; i < L; ++i)
            if (i < rev_[i]) std::swap(a[i], a[rev_[i]]);
        for (size_t len = 2; len <= L; len <<= 1) {
            const size_t half = len / 2, step = L / len;
            for (size_t i = 0; i < L; i += len)
                for (size_t k = 0; k < half; ++k) {
                    cplx w = inverse ? std::conj(tw_[k * step]) : tw_[k * step];
                    cplx u = a[i + k], v = a[i + k + half] * w;
                    a[i + k] = u + v;
                    a[i + k + half] = u - v;
                }
        }
    }
};

// One 1-D transform of x[0..n). The inverse is conj(fft(conj(x))), without the 1/n scaling.
// Every version (sequential, naive, designed) goes through this function, so a line gets
// exactly the same arithmetic whichever thread runs it.
inline void transform_line(const FftPlan& p, cplx* x, size_t n, bool inverse, cplx* work) {
    if (inverse)
        for (size_t i = 0; i < n; ++i) x[i] = std::conj(x[i]);
    p.forward(x, work);
    if (inverse)
        for (size_t i = 0; i < n; ++i) x[i] = std::conj(x[i]);
}

// Rows transformed together by one chunk of the row pass (see rows_pass).
constexpr long kRowTile = 8;

// Column pass of the designed version: load(c, column) fills column c of o (it may do
// nothing if o already holds the data), then the column is transformed in place.
// Columns are independent of each other and contiguous in memory. Each chunk owns a
// contiguous range of whole columns and its own workspace, allocated once per chunk; the
// plan is only read. Uniform work, so a static block partition.
template <class Load>
void columns_pass(CMat& o, const FftPlan& plan, bool inverse, const Config& cfg, Load&& load) {
    const size_t R = o.rows;
    par_for(cfg, o.cols, Sched{}, [&](long lo, long hi) {
        std::vector<cplx> work(plan.work_size());
        for (long c = lo; c < hi; ++c) {
            cplx* col = &o.d[static_cast<size_t>(c) * R];
            load(c, col);
            transform_line(plan, col, R, inverse, work.data());
        }
    });
}

// Row pass of the designed version: every row of o is transformed and handed to
// store(r, c, value). Rows are independent of each other, but a row is strided in a
// column-major matrix and neighbouring rows share cache lines. Each chunk therefore owns a
// contiguous block of rows and walks it in tiles of kRowTile rows: a tile is gathered
// column by column (kRowTile adjacent elements per column, a few sequential write streams)
// into a row-major scratch, transformed there, and scattered the same way. Only the rows
// at the border between two chunks share cache lines, and each element is written once.
// The scratch and the workspace are allocated once per chunk.
template <class Store>
void rows_pass(const CMat& o, const FftPlan& plan, bool inverse, const Config& cfg, Store&& store) {
    const size_t R = o.rows, C = o.cols;
    par_for(cfg, o.rows, Sched{}, [&](long lo, long hi) {
        std::vector<cplx> tile(static_cast<size_t>(kRowTile) * C), work(plan.work_size());
        for (long r0 = lo; r0 < hi; r0 += kRowTile) {
            const long nt = std::min(kRowTile, hi - r0);
            for (size_t c = 0; c < C; ++c) {
                const cplx* src = &o.d[c * R + r0];
                for (long t = 0; t < nt; ++t) tile[t * C + c] = src[t];
            }
            for (long t = 0; t < nt; ++t) transform_line(plan, &tile[t * C], C, inverse, work.data());
            for (size_t c = 0; c < C; ++c)
                for (long t = 0; t < nt; ++t) store(static_cast<int>(r0 + t), static_cast<int>(c), tile[t * C + c]);
        }
    });
}

// Transform every column (dim 1) then every row (dim 2): sequential baseline, and the naive
// version, which puts a `parallel for` on both loops with a private copy of the buffers
// per thread.
void fft2_inplace(CMat& o, bool inverse, const Config& cfg) {
    (void)cfg;
    const FftPlan pr(o.rows), pc(o.cols);
    std::vector<cplx> buf(std::max(o.rows, o.cols));
    std::vector<cplx> work(std::max(pr.work_size(), pc.work_size()));
    #pragma omp parallel for num_threads(cfg.threads) if(cfg.naive()) firstprivate(work)
    for (int c = 0; c < o.cols; ++c)
        transform_line(pr, &o.d[static_cast<size_t>(c) * o.rows], o.rows, inverse, work.data());
    #pragma omp parallel for num_threads(cfg.threads) if(cfg.naive()) firstprivate(buf, work)
    for (int r = 0; r < o.rows; ++r) {
        for (int c = 0; c < o.cols; ++c) buf[c] = o(r, c);
        transform_line(pc, buf.data(), o.cols, inverse, work.data());
        for (int c = 0; c < o.cols; ++c) o(r, c) = buf[c];
    }
}
}  // namespace

void fft_inplace(std::vector<cplx>& x, bool inverse) {
    if (x.size() <= 1) return;
    FftPlan p(x.size());
    std::vector<cplx> work(p.work_size());
    if (inverse)
        for (cplx& v : x) v = std::conj(v);
    p.forward(x.data(), work.data());
    if (inverse)
        for (cplx& v : x) v = std::conj(v);
}

CMat fft2(const Mat& a, const Config& cfg) {
    CMat o(a.rows, a.cols);
    if (cfg.designed()) {
        // the real-to-complex copy is done by the chunk that transforms the column
        const FftPlan pr(a.rows), pc(a.cols);
        columns_pass(o, pr, false, cfg, [&](long c, cplx* col) {
            const double* src = &a.d[static_cast<size_t>(c) * a.rows];
            for (int r = 0; r < a.rows; ++r) col[r] = src[r];
        });
        rows_pass(o, pc, false, cfg, [&](int r, int c, const cplx& v) { o(r, c) = v; });
        return o;
    }
    for (size_t i = 0; i < a.numel(); ++i) o.d[i] = a.d[i];
    fft2_inplace(o, false, cfg);
    return o;
}

CMat ifft2(const CMat& a, const Config& cfg) {
    const double s = 1.0 / (static_cast<double>(a.rows) * a.cols);
    if (cfg.designed()) {
        // copy and scaling are folded into the two passes: no serial sweep over the matrix
        CMat o(a.rows, a.cols);
        const FftPlan pr(a.rows), pc(a.cols);
        columns_pass(o, pr, true, cfg, [&](long c, cplx* col) {
            const cplx* src = &a.d[static_cast<size_t>(c) * a.rows];
            std::copy(src, src + a.rows, col);
        });
        rows_pass(o, pc, true, cfg, [&](int r, int c, cplx v) {
            v *= s;
            o(r, c) = v;
        });
        return o;
    }
    CMat o = a;
    fft2_inplace(o, true, cfg);
    for (cplx& v : o.d) v *= s;
    return o;
}

Mat ifft2_real(CMat a, const Config& cfg) {
    const double s = 1.0 / (static_cast<double>(a.rows) * a.cols);
    Mat o(a.rows, a.cols);
    if (cfg.designed()) {
        const FftPlan pr(a.rows), pc(a.cols);
        columns_pass(a, pr, true, cfg, [](long, cplx*) {});
        // chunks own disjoint blocks of rows of the output
        rows_pass(a, pc, true, cfg, [&](int r, int c, cplx v) {
            v *= s;
            o(r, c) = v.real();
        });
        return o;
    }
    fft2_inplace(a, true, cfg);
    for (size_t i = 0; i < a.numel(); ++i) {
        a.d[i] *= s;
        o.d[i] = a.d[i].real();
    }
    return o;
}

namespace {
// o(r, c) = a((r + pr) mod rows, (c + pc) mod cols). Pure permutation: every output column
// is written by exactly one chunk and the input is only read.
CMat shift2(const CMat& a, int pr, int pc, const Config& cfg) {
    CMat o(a.rows, a.cols);
    par_for(cfg, a.cols, Sched{}, [&](long lo, long hi) {
        for (long c = lo; c < hi; ++c) {
            const cplx* src = &a.d[static_cast<size_t>((c + pc) % a.cols) * a.rows];
            cplx* dst = &o.d[static_cast<size_t>(c) * a.rows];
            std::copy(src + pr, src + a.rows, dst);
            std::copy(src, src + pr, dst + (a.rows - pr));
        }
    });
    return o;
}

CMat shift2(const CMat& a, int pr, int pc) {
    CMat o(a.rows, a.cols);
    for (int c = 0; c < a.cols; ++c)
        for (int r = 0; r < a.rows; ++r) o(r, c) = a((r + pr) % a.rows, (c + pc) % a.cols);
    return o;
}
}  // namespace

CMat fftshift(const CMat& a) { return shift2(a, (a.rows + 1) / 2, (a.cols + 1) / 2); }
CMat ifftshift(const CMat& a) { return shift2(a, a.rows / 2, a.cols / 2); }
CMat fftshift(const CMat& a, const Config& cfg) { return shift2(a, (a.rows + 1) / 2, (a.cols + 1) / 2, cfg); }
CMat ifftshift(const CMat& a, const Config& cfg) { return shift2(a, a.rows / 2, a.cols / 2, cfg); }

}  // namespace mc
