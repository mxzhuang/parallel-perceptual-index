#include "matlab_compat.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace mc {

namespace {
constexpr double kPi = 3.14159265358979323846;

inline double matlab_round(double x) {  // round half away from zero
    return x >= 0 ? std::floor(x + 0.5) : -std::floor(-x + 0.5);
}
inline double clamp_u8(double x) { return std::min(255.0, std::max(0.0, x)); }
inline int pos_mod(long long a, long long m) {
    long long r = a % m;
    return static_cast<int>(r < 0 ? r + m : r);
}
}  // namespace

// ---- colour / image ---------------------------------------------------------------

Mat rgb2ycbcr_y(const uint8_t* rgb, int rows, int cols) {
    // MATLAB: T = (1/255) * [65.481 128.553 24.966], offset 16, imlincomb rounds to uint8.
    const double s = 1.0 / 255.0;
    const double c0 = s * 65.481, c1 = s * 128.553, c2 = s * 24.966;
    Mat y(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const uint8_t* p = rgb + (static_cast<size_t>(r) * cols + c) * 3;
            double v = c0 * p[0] + c1 * p[1] + c2 * p[2] + 16.0;
            y(r, c) = clamp_u8(matlab_round(v));
        }
    return y;
}

Mat rgb2gray(const uint8_t* rgb, int rows, int cols) {
    const double c0 = 0.298936021293775, c1 = 0.587043074451121, c2 = 0.114020904255103;
    Mat g(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            const uint8_t* p = rgb + (static_cast<size_t>(r) * cols + c) * 3;
            g(r, c) = clamp_u8(matlab_round(c0 * p[0] + c1 * p[1] + c2 * p[2]));
        }
    return g;
}

Mat shave(const Mat& a, int w) {
    if (w <= 0) return a;
    return crop(a, w, w, a.rows - 2 * w, a.cols - 2 * w);
}

Mat crop(const Mat& a, int r0, int c0, int nr, int nc) {
    Mat o(nr, nc);
    for (int c = 0; c < nc; ++c)
        for (int r = 0; r < nr; ++r) o(r, c) = a(r0 + r, c0 + c);
    return o;
}

// ---- filtering ----------------------------------------------------------------------

Mat fspecial_gaussian(int n, double sigma) {
    const double siz = (n - 1) / 2.0;
    Mat h(n, n);
    double mx = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            double x = -siz + c, y = -siz + r;  // meshgrid: x varies along columns
            double arg = -(x * x + y * y) / (2 * sigma * sigma);
            h(r, c) = std::exp(arg);
            mx = std::max(mx, h(r, c));
        }
    double sum = 0;
    for (double& v : h.d) {
        if (v < DBL_EPSILON * mx) v = 0;
        sum += v;
    }
    if (sum != 0)
        for (double& v : h.d) v /= sum;
    return h;
}

Vec gaussian_1d(int n, double sigma) {
    const double siz = (n - 1) / 2.0;
    Vec k(n);
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        double x = -siz + i;
        k[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += k[i];
    }
    for (double& v : k) v /= sum;
    return k;
}

Mat imfilter(const Mat& a, const Mat& h, Boundary b) {
    const int R = a.rows, C = a.cols;
    const int cr = (h.rows + 1) / 2 - 1, cc = (h.cols + 1) / 2 - 1;
    Mat o(R, C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int j = 0; j < h.cols; ++j) {
                int cj = c + j - cc;
                if (b == Boundary::Zero && (cj < 0 || cj >= C)) continue;
                cj = std::min(C - 1, std::max(0, cj));
                for (int i = 0; i < h.rows; ++i) {
                    int ri = r + i - cr;
                    if (b == Boundary::Zero && (ri < 0 || ri >= R)) continue;
                    ri = std::min(R - 1, std::max(0, ri));
                    s += h(i, j) * a(ri, cj);
                }
            }
            o(r, c) = s;
        }
    return o;
}

Mat filter2_valid(const Mat& h, const Mat& a) {
    const int R = a.rows - h.rows + 1, C = a.cols - h.cols + 1;
    if (R <= 0 || C <= 0) return Mat();
    Mat o(R, C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int j = 0; j < h.cols; ++j)
                for (int i = 0; i < h.rows; ++i) s += h(i, j) * a(r + i, c + j);
            o(r, c) = s;
        }
    return o;
}

Mat filter2_valid_separable(const Vec& k, const Mat& a) {
    const int n = static_cast<int>(k.size());
    const int R = a.rows - n + 1, C = a.cols - n + 1;
    if (R <= 0 || C <= 0) return Mat();
    Mat t(R, a.cols);  // vertical pass
    for (int c = 0; c < a.cols; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int i = 0; i < n; ++i) s += k[i] * a(r + i, c);
            t(r, c) = s;
        }
    Mat o(R, C);  // horizontal pass
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) {
            double s = 0;
            for (int j = 0; j < n; ++j) s += k[j] * t(r, c + j);
            o(r, c) = s;
        }
    return o;
}

Mat downsample2(const Mat& a) {
    Mat o(a.rows / 2, a.cols / 2);
    for (int c = 0; c < o.cols; ++c)
        for (int r = 0; r < o.rows; ++r) o(r, c) = a(2 * r + 1, 2 * c + 1);
    return o;
}

Mat im2col_distinct(const Mat& a, int m, int n) {
    const int nbr = (a.rows + m - 1) / m, nbc = (a.cols + n - 1) / n;
    Mat o(m * n, nbr * nbc);
    for (int bc = 0; bc < nbc; ++bc)
        for (int br = 0; br < nbr; ++br) {
            int col = br + bc * nbr;
            for (int cc = 0; cc < n; ++cc)
                for (int rr = 0; rr < m; ++rr) {
                    int r = br * m + rr, c = bc * n + cc;
                    o(cc * m + rr, col) = (r < a.rows && c < a.cols) ? a(r, c) : 0.0;
                }
        }
    return o;
}

namespace {
double cubic(double x) {
    double ax = std::fabs(x), ax2 = ax * ax, ax3 = ax2 * ax;
    if (ax <= 1) return 1.5 * ax3 - 2.5 * ax2 + 1;
    if (ax <= 2) return -0.5 * ax3 + 2.5 * ax2 - 4 * ax + 2;
    return 0;
}

struct Contrib {
    int out_len = 0, P = 0;
    std::vector<double> w;  // out_len x P, row-major
    std::vector<int> idx;   // 0-based input index
};

// MATLAB imresize>contributions (bicubic, antialiasing on)
Contrib contributions(int in_len, int out_len, double scale) {
    double kw = 4.0;
    const bool aa = scale < 1;
    if (aa) kw = kw / scale;
    const int P = static_cast<int>(std::ceil(kw)) + 2;
    std::vector<double> w(static_cast<size_t>(out_len) * P);
    std::vector<int> ind(static_cast<size_t>(out_len) * P);
    for (int i = 0; i < out_len; ++i) {
        double x = i + 1;
        double u = x / scale + 0.5 * (1 - 1 / scale);
        double left = std::floor(u - kw / 2);
        double sum = 0;
        for (int j = 0; j < P; ++j) {
            double index = left + j;
            double dist = u - index;
            double v = aa ? scale * cubic(scale * dist) : cubic(dist);
            w[i * P + j] = v;
            sum += v;
            int i1 = static_cast<int>(index);  // 1-based, may be out of range
            int m = pos_mod(static_cast<long long>(i1) - 1, 2LL * in_len);
            ind[i * P + j] = m < in_len ? m : 2 * in_len - 1 - m;  // aux = [1:n n:-1:1]
        }
        for (int j = 0; j < P; ++j) w[i * P + j] /= sum;
    }
    // drop columns whose weights are all zero
    std::vector<int> keep;
    for (int j = 0; j < P; ++j) {
        bool any = false;
        for (int i = 0; i < out_len && !any; ++i) any = w[i * P + j] != 0;
        if (any) keep.push_back(j);
    }
    Contrib ct;
    ct.out_len = out_len;
    ct.P = static_cast<int>(keep.size());
    ct.w.resize(static_cast<size_t>(out_len) * ct.P);
    ct.idx.resize(static_cast<size_t>(out_len) * ct.P);
    for (int i = 0; i < out_len; ++i)
        for (int jj = 0; jj < ct.P; ++jj) {
            ct.w[i * ct.P + jj] = w[i * P + keep[jj]];
            ct.idx[i * ct.P + jj] = ind[i * P + keep[jj]];
        }
    return ct;
}

Mat resize_rows(const Mat& a, int out_rows, double scale) {
    Contrib ct = contributions(a.rows, out_rows, scale);
    Mat o(out_rows, a.cols);
    for (int c = 0; c < a.cols; ++c)
        for (int i = 0; i < out_rows; ++i) {
            double s = 0;
            for (int k = 0; k < ct.P; ++k) s += ct.w[i * ct.P + k] * a(ct.idx[i * ct.P + k], c);
            o(i, c) = s;
        }
    return o;
}

Mat resize_cols(const Mat& a, int out_cols, double scale) {
    Contrib ct = contributions(a.cols, out_cols, scale);
    Mat o(a.rows, out_cols);
    for (int i = 0; i < out_cols; ++i)
        for (int r = 0; r < a.rows; ++r) {
            double s = 0;
            for (int k = 0; k < ct.P; ++k) s += ct.w[i * ct.P + k] * a(r, ct.idx[i * ct.P + k]);
            o(r, i) = s;
        }
    return o;
}

Mat resize_impl(const Mat& a, int out_r, int out_c, double sr, double sc) {
    // MATLAB processes dimensions in ascending order of scale (stable sort)
    if (sr <= sc) return resize_cols(resize_rows(a, out_r, sr), out_c, sc);
    return resize_rows(resize_cols(a, out_c, sc), out_r, sr);
}
}  // namespace

Mat imresize_scale(const Mat& a, double scale) {
    int out_r = static_cast<int>(std::ceil(scale * a.rows));
    int out_c = static_cast<int>(std::ceil(scale * a.cols));
    return resize_impl(a, out_r, out_c, scale, scale);
}

Mat imresize_size(const Mat& a, int out_rows, int out_cols) {
    return resize_impl(a, out_rows, out_cols, static_cast<double>(out_rows) / a.rows,
                       static_cast<double>(out_cols) / a.cols);
}

Mat circshift(const Mat& a, int dr, int dc) {
    Mat o(a.rows, a.cols);
    for (int c = 0; c < a.cols; ++c) {
        int sc = pos_mod(static_cast<long long>(c) - dc, a.cols);
        for (int r = 0; r < a.rows; ++r) o(r, c) = a(pos_mod(static_cast<long long>(r) - dr, a.rows), sc);
    }
    return o;
}

// ---- transforms ---------------------------------------------------------------------

Dct2::Dct2(int n) : n_(n), c_(static_cast<size_t>(n) * n) {
    for (int k = 0; k < n; ++k) {
        double a = k == 0 ? std::sqrt(1.0 / n) : std::sqrt(2.0 / n);
        for (int x = 0; x < n; ++x) c_[static_cast<size_t>(x) * n + k] = a * std::cos(kPi * (2 * x + 1) * k / (2.0 * n));
    }
}

void Dct2::apply(const double* in, double* out) const {
    const int n = n_;
    double t[64];  // n <= 8
    // t = C * in
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += c_[static_cast<size_t>(k) * n + r] * in[c * n + k];
            t[c * n + r] = s;
        }
    // out = t * C'
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            double s = 0;
            for (int k = 0; k < n; ++k) s += t[k * n + r] * c_[static_cast<size_t>(k) * n + c];
            out[c * n + r] = s;
        }
}

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
void fft2_inplace(CMat& o, bool inverse) {
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

CMat fft2(const Mat& a) {
    CMat o(a.rows, a.cols);
    for (size_t i = 0; i < a.numel(); ++i) o.d[i] = a.d[i];
    fft2_inplace(o, false);
    return o;
}

CMat ifft2(const CMat& a) {
    CMat o = a;
    fft2_inplace(o, true);
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

// ---- small dense linear algebra ---------------------------------------------------

void eig_sym(const Mat& a0, Vec& vals, Mat& vecs) {
    const int n = a0.rows;
    Mat a = a0;
    vecs = Mat(n, n);
    for (int i = 0; i < n; ++i) vecs(i, i) = 1;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0, diag = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) (i == j ? diag : off) += a(i, j) * a(i, j);
        if (off <= 1e-30 * diag || off == 0) break;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                double apq = a(p, q);
                if (apq == 0) continue;
                double theta = (a(q, q) - a(p, p)) / (2 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                double cs = 1 / std::sqrt(t * t + 1), sn = t * cs;
                for (int k = 0; k < n; ++k) {  // A = A * J
                    double akp = a(k, p), akq = a(k, q);
                    a(k, p) = cs * akp - sn * akq;
                    a(k, q) = sn * akp + cs * akq;
                }
                for (int k = 0; k < n; ++k) {  // A = J' * A
                    double apk = a(p, k), aqk = a(q, k);
                    a(p, k) = cs * apk - sn * aqk;
                    a(q, k) = sn * apk + cs * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double vkp = vecs(k, p), vkq = vecs(k, q);
                    vecs(k, p) = cs * vkp - sn * vkq;
                    vecs(k, q) = sn * vkp + cs * vkq;
                }
            }
    }
    vals.assign(n, 0);
    for (int i = 0; i < n; ++i) vals[i] = a(i, i);
}

Mat pinv_sym(const Mat& a) {
    const int n = a.rows;
    Vec lam;
    Mat v;
    eig_sym(a, lam, v);
    double nrm = 0;
    for (double l : lam) nrm = std::max(nrm, std::fabs(l));
    const double tol = n * (std::nextafter(nrm, INFINITY) - nrm);  // max(size) * eps(norm)
    Mat p(n, n);
    for (int k = 0; k < n; ++k) {
        if (std::fabs(lam[k]) <= tol) continue;
        const double inv = 1.0 / lam[k];
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) p(i, j) += v(i, k) * inv * v(j, k);
    }
    return p;
}

Vec singular_values_wide(const Mat& a) {
    // svd(A) for A (m x n), m small: Householder QR of A' (n x m), then one-sided Jacobi on R.
    const int m = a.rows, n = a.cols;
    Mat b(n, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i) b(i, j) = a(j, i);
    const int k_end = std::min(m, n);
    for (int k = 0; k < k_end; ++k) {
        double norm = 0;
        for (int i = k; i < n; ++i) norm += b(i, k) * b(i, k);
        norm = std::sqrt(norm);
        if (norm == 0) continue;
        double alpha = b(k, k) > 0 ? -norm : norm;
        std::vector<double> v(n - k);
        for (int i = k; i < n; ++i) v[i - k] = b(i, k);
        v[0] -= alpha;
        double vn = 0;
        for (double x : v) vn += x * x;
        if (vn == 0) continue;
        for (int j = k; j < m; ++j) {
            double dot = 0;
            for (int i = k; i < n; ++i) dot += v[i - k] * b(i, j);
            double f = 2 * dot / vn;
            for (int i = k; i < n; ++i) b(i, j) -= f * v[i - k];
        }
    }
    const int r = k_end;
    Mat R(r, m);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i <= std::min(j, r - 1); ++i) R(i, j) = b(i, j);
    // one-sided Jacobi (Hestenes) on the columns of R
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
                double zeta = (beta - alpha) / (2 * gamma);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                double cs = 1 / std::sqrt(1 + t * t), sn = cs * t;
                for (int i = 0; i < r; ++i) {
                    double x = R(i, p), y = R(i, q);
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
    sv.resize(std::min(m, n));
    return sv;
}

// ---- statistics ---------------------------------------------------------------------

double mean(const double* x, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s / static_cast<double>(n);
}

double var(const double* x, size_t n) {
    if (n < 2) return 0;
    double mu = mean(x, n), s = 0;
    for (size_t i = 0; i < n; ++i) s += (x[i] - mu) * (x[i] - mu);
    return s / static_cast<double>(n - 1);
}

double stdev(const double* x, size_t n) { return std::sqrt(var(x, n)); }

Vec colon(double a, double d, double b) {
    const long long n = static_cast<long long>(std::floor((b - a) / d + 1e-10));
    Vec v(static_cast<size_t>(n + 1));
    const double last = a + n * d;
    const long long half = n / 2;
    for (long long k = 0; k <= n; ++k) v[k] = k <= half ? a + k * d : last - (n - k) * d;
    return v;
}

}  // namespace mc
