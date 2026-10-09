#include "ma.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "ggd.h"
#include "matlab_compat.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

// ============================================================================
// M1: spatial pyramid (feature_all.m)
// ============================================================================
struct SpatialPyramid {
    Mat im[3];  // full, half, quarter resolution, values in [0, 1]
};

SpatialPyramid spatial_pyramid(const Mat& gray) {
    SpatialPyramid p;
    p.im[0] = gray;
    for (double& v : p.im[0].d) v /= 255.0;  // im2double
    const Mat h = mc::fspecial_gaussian(3, 0.5);
    p.im[1] = mc::downsample2(mc::imfilter(p.im[0], h));
    p.im[2] = mc::downsample2(mc::imfilter(p.im[1], h));
    return p;
}

// ============================================================================
// M2: block-DCT statistics (block_dct.m and the per-block functions)
// ============================================================================
double std_over_mean(const double* a, size_t n, double eps) {
    return mc::stdev(a, n) / (mc::mean(a, n) + eps);
}

// mean of sorted values in [from, to)
double mean_range(const Vec& s, size_t from, size_t to) { return mc::mean(s.data() + from, to - from); }

void block_dct(const Mat& im, double out[6]) {
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
            dct.apply(win, D);  // computed once, shared by all five statistics
            const size_t k = br + static_cast<size_t>(bc) * nbr;

            gama[k] = ggd.fit(D + 1, 48);  // all coefficients except DC (column-major)
            for (int i = 0; i < 48; ++i) a48[i] = std::fabs(D[i + 1]);
            cv[k] = std_over_mean(a48, 48, 0.0000001);

            double g[3];
            const int (*lists[3])[2] = {o1, o2, o3};
            for (int l = 0; l < 3; ++l) {
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

// ============================================================================
// M3: patch SVD (svd(im2col(im, [5 5], 'distinct')))
// ============================================================================
Vec patch_svd(const Mat& im) { return mc::singular_values_wide(mc::im2col_distinct(im, 5, 5)); }

// ============================================================================
// M4: steerable pyramid in the frequency domain (matlabPyrTools buildSFpyr / buildSFpyrLevs)
// ============================================================================
struct Lut {
    Vec y;
    double origin, increment;
};

// pointOp MEX: linear interpolation in a lookup table, linear extrapolation at both ends
double point_op(const Lut& lut, double v) {
    const int maxidx = static_cast<int>(lut.y.size()) - 2;
    double pos = (v - lut.origin) / lut.increment;
    int index = static_cast<int>(pos);  // C cast truncates, as in the MEX
    if (index < 0)
        index = 0;
    else if (index > maxidx)
        index = maxidx;
    return lut.y[index] + (lut.y[index + 1] - lut.y[index]) * (pos - index);
}

Mat point_op(const Mat& im, const Lut& lut) {
    Mat o(im.rows, im.cols);
    for (size_t i = 0; i < im.numel(); ++i) o.d[i] = point_op(lut, im.d[i]);
    return o;
}

struct RcosFn {
    Vec X, Y;
};

// rcosFn(width = 1, position = -0.5, values = [0 1])
RcosFn rcos_fn() {
    const int sz = 256;
    RcosFn f;
    f.X.resize(sz + 3);
    f.Y.resize(sz + 3);
    for (int j = 0; j < sz + 3; ++j) {
        double x = (kPi * (j - sz - 1)) / (2 * sz);
        double c = std::cos(x);
        f.Y[j] = 0 + 1 * (c * c);
        f.X[j] = x;
    }
    f.Y[0] = f.Y[1];
    f.Y[sz + 2] = f.Y[sz + 1];
    for (double& x : f.X) x = -0.5 + (2 * 1 / kPi) * (x + kPi / 4);
    return f;
}

void sf_levels(CMat lodft, Mat log_rad, Mat angle, Vec Xrcos, const Vec& Yrcos, int ht, int level,
               SteerablePyramid& pyr) {
    if (ht <= 0) return;  // the final low-pass residual is not used by Ma's features
    const int nbands = 6, order = 5;
    for (double& x : Xrcos) x -= 1.0;  // log2(2)
    const int lutsize = 1024;
    Lut angle_lut;
    angle_lut.y.resize(3 * lutsize + 3);
    Vec Xcosn(3 * lutsize + 3);
    for (int j = 0; j < 3 * lutsize + 3; ++j) Xcosn[j] = (kPi * (j - (2 * lutsize + 1))) / lutsize;
    const double cnst = (std::pow(2.0, 2 * order) * (120.0 * 120.0)) / (nbands * 3628800.0);
    for (int j = 0; j < 3 * lutsize + 3; ++j) angle_lut.y[j] = std::sqrt(cnst) * std::pow(std::cos(Xcosn[j]), order);
    angle_lut.increment = Xcosn[1] - Xcosn[0];

    const Lut hilut{Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0]};
    const Mat himask = point_op(log_rad, hilut);
    const cplx rot(0.0, -1.0);  // (-sqrt(-1))^5
    for (int b = 0; b < nbands; ++b) {
        angle_lut.origin = Xcosn[0] + kPi * b / nbands;
        const Mat anglemask = point_op(angle, angle_lut);
        CMat banddft(lodft.rows, lodft.cols);
        for (size_t i = 0; i < lodft.numel(); ++i) banddft.d[i] = rot * lodft.d[i] * anglemask.d[i] * himask.d[i];
        CMat band = mc::ifft2(mc::ifftshift(banddft));
        Mat re(band.rows, band.cols);
        for (size_t i = 0; i < band.numel(); ++i) re.d[i] = band.d[i].real();
        pyr.band[level][b] = std::move(re);
    }
    if (ht - 1 <= 0) return;

    const int R = lodft.rows, C = lodft.cols;
    const int ctr[2] = {static_cast<int>(std::ceil((R + 0.5) / 2)), static_cast<int>(std::ceil((C + 0.5) / 2))};
    const int lod[2] = {static_cast<int>(std::ceil((R - 0.5) / 2)), static_cast<int>(std::ceil((C - 0.5) / 2))};
    const int loctr[2] = {static_cast<int>(std::ceil((lod[0] + 0.5) / 2)), static_cast<int>(std::ceil((lod[1] + 0.5) / 2))};
    const int r0 = ctr[0] - loctr[0], c0 = ctr[1] - loctr[1];  // lostart - 1 (0-based)
    log_rad = mc::crop(log_rad, r0, c0, lod[0], lod[1]);
    angle = mc::crop(angle, r0, c0, lod[0], lod[1]);
    CMat lo(lod[0], lod[1]);
    for (int c = 0; c < lod[1]; ++c)
        for (int r = 0; r < lod[0]; ++r) lo(r, c) = lodft(r0 + r, c0 + c);
    Vec YIrcos(Yrcos.size());
    for (size_t i = 0; i < Yrcos.size(); ++i) YIrcos[i] = std::fabs(std::sqrt(1.0 - Yrcos[i] * Yrcos[i]));
    const Mat lomask = point_op(log_rad, Lut{YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0]});
    for (size_t i = 0; i < lo.numel(); ++i) lo.d[i] = lomask.d[i] * lo.d[i];
    sf_levels(std::move(lo), std::move(log_rad), std::move(angle), Xrcos, Yrcos, ht - 1, level + 1, pyr);
}
}  // namespace

SteerablePyramid build_sf_pyramid(const Mat& im) {
    const int R = im.rows, C = im.cols;
    const int ht = 2;
    if (ht > static_cast<int>(std::floor(std::log2(std::min(R, C)))) - 2)
        throw std::runtime_error("image too small for a 2-level steerable pyramid");
    const int ctr1 = static_cast<int>(std::ceil((R + 0.5) / 2)), ctr2 = static_cast<int>(std::ceil((C + 0.5) / 2));
    Mat log_rad(R, C), angle(R, C);
    for (int c = 0; c < C; ++c)
        for (int r = 0; r < R; ++r) {
            double x = ((c + 1) - ctr2) / (C / 2.0);
            double y = ((r + 1) - ctr1) / (R / 2.0);
            angle(r, c) = std::atan2(y, x);
            log_rad(r, c) = std::sqrt(x * x + y * y);
        }
    log_rad(ctr1 - 1, ctr2 - 1) = log_rad(ctr1 - 1, ctr2 - 2);
    for (double& v : log_rad.d) v = std::log2(v);

    RcosFn rc = rcos_fn();
    Vec Yrcos(rc.Y.size()), YIrcos(rc.Y.size());
    for (size_t i = 0; i < rc.Y.size(); ++i) {
        Yrcos[i] = std::sqrt(rc.Y[i]);
        YIrcos[i] = std::sqrt(1.0 - Yrcos[i] * Yrcos[i]);
    }
    const double inc = rc.X[1] - rc.X[0];
    const Mat lo0mask = point_op(log_rad, Lut{YIrcos, rc.X[0], inc});
    const Mat hi0mask = point_op(log_rad, Lut{Yrcos, rc.X[0], inc});

    const CMat imdft = mc::fftshift(mc::fft2(im));
    CMat lo0dft(R, C), hi0dft(R, C);
    for (size_t i = 0; i < imdft.numel(); ++i) {
        lo0dft.d[i] = imdft.d[i] * lo0mask.d[i];
        hi0dft.d[i] = imdft.d[i] * hi0mask.d[i];
    }
    SteerablePyramid pyr;
    sf_levels(std::move(lo0dft), log_rad, angle, rc.X, Yrcos, ht, 0, pyr);
    CMat hi0 = mc::ifft2(mc::ifftshift(hi0dft));
    pyr.hi0 = Mat(R, C);
    for (size_t i = 0; i < hi0.numel(); ++i) pyr.hi0.d[i] = hi0.d[i].real();
    return pyr;
}

namespace {
// ============================================================================
// M5: divisive normalisation (norm_sender_normalized.m, Nsc = 2, Nor = 6, 3x3 block,
//     parent and neighbours on). The unused histogram of the original is omitted.
// ============================================================================
struct NormalizedBands {
    Vec subband[12];     // vectorised, guard band removed, zero mean
    int size_band[12][2];
};

NormalizedBands divisive_normalization(const SteerablePyramid& pyr) {
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
double ssim_structure(const Mat& a, const Mat& b, ConvMode conv) {
    const SsimWindow& w = ssim_window();
    auto filt = [&](const Mat& x) {
        return conv == ConvMode::Direct ? mc::filter2_valid(w.w2d, x) : mc::filter2_valid_separable(w.w1d, x);
    };
    Mat aa(a.rows, a.cols), bb(a.rows, a.cols), ab(a.rows, a.cols);
    for (size_t i = 0; i < a.numel(); ++i) {
        aa.d[i] = a.d[i] * a.d[i];
        bb.d[i] = b.d[i] * b.d[i];
        ab.d[i] = a.d[i] * b.d[i];
    }
    const Mat mu1 = filt(a), mu2 = filt(b), s11 = filt(aa), s22 = filt(bb), s12 = filt(ab);
    const double C2 = (0.03 * 255) * (0.03 * 255);
    double sum = 0;
    for (size_t i = 0; i < mu1.numel(); ++i) {
        double m1 = mu1.d[i], m2 = mu2.d[i];
        double sigma1_sq = s11.d[i] - m1 * m1;
        double sigma2_sq = s22.d[i] - m2 * m2;
        double sigma12 = s12.d[i] - m1 * m2;
        sum += (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2);
    }
    return sum / static_cast<double>(mu1.numel());
}

Mat reshape(const Vec& v, int rows, int cols) {
    Mat m(rows, cols);
    m.d = v;
    return m;
}
}  // namespace

MaFeatures ma_features(const Mat& gray, ConvMode conv, ModuleTimer* timer) {
    MaFeatures f;
    SpatialPyramid sp;
    {
        ModuleTimer::Scope t(timer, 1);
        sp = spatial_pyramid(gray);
    }
    {
        ModuleTimer::Scope t(timer, 2);
        f.f1.resize(18);
        for (int s = 0; s < 3; ++s) block_dct(sp.im[s], &f.f1[6 * s]);
    }
    {
        ModuleTimer::Scope t(timer, 3);
        for (int s = 0; s < 3; ++s) {
            Vec sv = patch_svd(sp.im[s]);
            f.f3.insert(f.f3.end(), sv.begin(), sv.end());
        }
    }
    SteerablePyramid pyr;
    {
        ModuleTimer::Scope t(timer, 4);
        pyr = build_sf_pyramid(gray);  // global_gsm works on double(img), 0..255
    }
    NormalizedBands nb;
    {
        ModuleTimer::Scope t(timer, 5);
        nb = divisive_normalization(pyr);
    }
    {
        ModuleTimer::Scope t(timer, 6);
        const GgdTable& ggd = ggd_table();
        f.f2.reserve(45);
        for (int i = 0; i < 12; ++i) f.f2.push_back(ggd.fit(nb.subband[i].data(), nb.subband[i].size()));
        for (int i = 0; i < 6; ++i) {
            Vec t2 = nb.subband[i];
            t2.insert(t2.end(), nb.subband[i + 6].begin(), nb.subband[i + 6].end());
            f.f2.push_back(ggd.fit(t2.data(), t2.size()));
        }
        for (int i = 0; i < 12; ++i) {
            const Mat& band = pyr.band[i / 6][i % 6];
            Mat up = mc::imresize_size(band, pyr.hi0.rows, pyr.hi0.cols);
            f.f2.push_back(ssim_structure(up, pyr.hi0, conv));
        }
        for (int i = 0; i < 6; ++i)
            for (int j = i + 1; j < 6; ++j)
                f.f2.push_back(ssim_structure(reshape(nb.subband[i], nb.size_band[i][0], nb.size_band[i][1]),
                                              reshape(nb.subband[j], nb.size_band[j][0], nb.size_band[j][1]), conv));
    }
    return f;
}
