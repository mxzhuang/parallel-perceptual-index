#include "fft.h"

#include <algorithm>
#include <cmath>

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

// transform every column (dim 1) then every row (dim 2)
void fft2_inplace(CMat& o, bool inverse, const Config& cfg) {
    (void)cfg;
    const FftPlan pr(o.rows), pc(o.cols);
    std::vector<cplx> buf(std::max(o.rows, o.cols));
    std::vector<cplx> work(std::max(pr.work_size(), pc.work_size()));
    auto run = [&](const FftPlan& p, cplx* x, size_t n) {
        if (inverse)
            for (size_t i = 0; i < n; ++i) x[i] = std::conj(x[i]);
        p.forward(x, work.data());
        if (inverse)
            for (size_t i = 0; i < n; ++i) x[i] = std::conj(x[i]);
    };
    for (int c = 0; c < o.cols; ++c) run(pr, &o.d[static_cast<size_t>(c) * o.rows], o.rows);
    for (int r = 0; r < o.rows; ++r) {
        for (int c = 0; c < o.cols; ++c) buf[c] = o(r, c);
        run(pc, buf.data(), o.cols);
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
    for (size_t i = 0; i < a.numel(); ++i) o.d[i] = a.d[i];
    fft2_inplace(o, false, cfg);
    return o;
}

CMat ifft2(const CMat& a, const Config& cfg) {
    CMat o = a;
    fft2_inplace(o, true, cfg);
    const double s = 1.0 / (static_cast<double>(a.rows) * a.cols);
    for (cplx& v : o.d) v *= s;
    return o;
}

namespace {
CMat shift2(const CMat& a, int pr, int pc) {
    CMat o(a.rows, a.cols);
    for (int c = 0; c < a.cols; ++c)
        for (int r = 0; r < a.rows; ++r) o(r, c) = a((r + pr) % a.rows, (c + pc) % a.cols);
    return o;
}
}  // namespace

CMat fftshift(const CMat& a) { return shift2(a, (a.rows + 1) / 2, (a.cols + 1) / 2); }
CMat ifftshift(const CMat& a) { return shift2(a, a.rows / 2, a.cols / 2); }

}  // namespace mc
