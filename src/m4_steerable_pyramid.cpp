#include "modules.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "fft.h"
#include "matlab_compat.h"
#include "parallel.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

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
               SteerablePyramid& pyr, const Config& cfg) {
    const bool faithful = cfg.variant == Variant::Faithful;
    if (ht <= 0) {
        // final low-pass residual: not used by Ma's features, computed only in the faithful variant
        if (faithful) {
            const CMat lo0 = mc::ifft2(mc::ifftshift(lodft), cfg);
            g_unused_sink = lo0.d[0].real();
        }
        return;
    }
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
        CMat band = mc::ifft2(mc::ifftshift(banddft), cfg);
        Mat re(band.rows, band.cols);
        for (size_t i = 0; i < band.numel(); ++i) re.d[i] = band.d[i].real();
        pyr.band[level][b] = std::move(re);
    }
    if (ht - 1 <= 0 && !faithful) return;

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
    sf_levels(std::move(lo), std::move(log_rad), std::move(angle), Xrcos, Yrcos, ht - 1, level + 1, pyr, cfg);
}
}  // namespace

PI_NOINLINE SteerablePyramid m4_steerable_pyramid(const Mat& im, const Config& cfg) {
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

    const CMat imdft = mc::fftshift(mc::fft2(im, cfg));
    CMat lo0dft(R, C), hi0dft(R, C);
    for (size_t i = 0; i < imdft.numel(); ++i) {
        lo0dft.d[i] = imdft.d[i] * lo0mask.d[i];
        hi0dft.d[i] = imdft.d[i] * hi0mask.d[i];
    }
    SteerablePyramid pyr;
    sf_levels(std::move(lo0dft), log_rad, angle, rc.X, Yrcos, ht, 0, pyr, cfg);
    CMat hi0 = mc::ifft2(mc::ifftshift(hi0dft), cfg);
    pyr.hi0 = Mat(R, C);
    for (size_t i = 0; i < hi0.numel(); ++i) pyr.hi0.d[i] = hi0.d[i].real();
    return pyr;
}
