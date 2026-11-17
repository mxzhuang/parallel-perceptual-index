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

constexpr int kBands = 6;  // orientations per scale

// Angular mask of an order-5 pyramid as a lookup table over [-2 pi, pi]; origin is the
// position of orientation 0, orientation b is shifted by pi * b / nbands.
Lut angle_table(int nbands) {
    const int order = 5;
    const int lutsize = 1024;
    Lut angle_lut;
    angle_lut.y.resize(3 * lutsize + 3);
    Vec Xcosn(3 * lutsize + 3);
    for (int j = 0; j < 3 * lutsize + 3; ++j) Xcosn[j] = (kPi * (j - (2 * lutsize + 1))) / lutsize;
    const double cnst = (std::pow(2.0, 2 * order) * (120.0 * 120.0)) / (nbands * 3628800.0);
    for (int j = 0; j < 3 * lutsize + 3; ++j) angle_lut.y[j] = std::sqrt(cnst) * std::pow(std::cos(Xcosn[j]), order);
    angle_lut.origin = Xcosn[0];
    angle_lut.increment = Xcosn[1] - Xcosn[0];
    return angle_lut;
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
    const int nbands = kBands;
    for (double& x : Xrcos) x -= 1.0;  // log2(2)
    Lut angle_lut = angle_table(nbands);
    const double angle_origin = angle_lut.origin;

    const Lut hilut{Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0]};
    const Mat himask = point_op(log_rad, hilut);
    const cplx rot(0.0, -1.0);  // (-sqrt(-1))^5
    for (int b = 0; b < nbands; ++b) {
        angle_lut.origin = angle_origin + kPi * b / nbands;
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

// ---- designed parallel version ---------------------------------------------------------
// Same arithmetic as the code above, element for element, so the output is bit-identical.
// The recursion is unrolled into its data flow:
//
//   forward FFT ─┬─ x high mask ───────────────────────────────> high-pass residual (1)
//                └─ x low mask ─┬─ x angle x radial masks ─────> scale-0 bands      (6)
//                               └─ crop x low mask ─ x masks ──> scale-1 bands      (6)
//
// Scale 1 needs only the low-pass spectrum, not the scale-0 bands, so once the two
// low-pass spectra exist the 13 inverse transforms are independent units.

// Geometry and masks of one scale, read-only while the bands are computed.
struct Scale {
    CMat lodft;         // low-pass spectrum, centred
    Mat log_rad, angle;
    Lut hilut;          // radial band-pass mask
};

// One oriented band: spectrum x angular mask x radial mask, then the inverse transform.
// The masks are evaluated on the fly instead of being stored as full-size matrices.
// Elements are independent; each chunk writes its own range of whole columns.
Mat oriented_band(const Scale& sc, const Lut& angle_lut, const Config& cfg) {
    const cplx rot(0.0, -1.0);  // (-sqrt(-1))^5
    const int R = sc.lodft.rows;
    CMat banddft(R, sc.lodft.cols);
    par_for(cfg, sc.lodft.cols, Sched{}, [&](long lo, long hi) {
        for (size_t i = static_cast<size_t>(lo) * R; i < static_cast<size_t>(hi) * R; ++i) {
            const double anglemask = point_op(angle_lut, sc.angle.d[i]);
            const double himask = point_op(sc.hilut, sc.log_rad.d[i]);
            banddft.d[i] = rot * sc.lodft.d[i] * anglemask * himask;
        }
    });
    return mc::ifft2_real(mc::ifftshift(banddft, cfg), cfg);
}

SteerablePyramid steerable_pyramid_designed(const Mat& im, const Config& cfg) {
    const int R = im.rows, C = im.cols;
    const int ctr1 = static_cast<int>(std::ceil((R + 0.5) / 2)), ctr2 = static_cast<int>(std::ceil((C + 0.5) / 2));
    Scale sc[2];

    // Polar coordinates of every frequency: independent per element, one pass. The centre
    // (radius 0) takes the radius of its left neighbour before the logarithm.
    sc[0].log_rad = Mat(R, C);
    sc[0].angle = Mat(R, C);
    par_for(cfg, C, Sched{}, [&](long lo, long hi) {
        for (int c = static_cast<int>(lo); c < hi; ++c)
            for (int r = 0; r < R; ++r) {
                double x = ((c + 1) - ctr2) / (C / 2.0);
                double y = ((r + 1) - ctr1) / (R / 2.0);
                sc[0].angle(r, c) = std::atan2(y, x);
                if (r == ctr1 - 1 && c == ctr2 - 1) x = (c - ctr2) / (C / 2.0);
                sc[0].log_rad(r, c) = std::log2(std::sqrt(x * x + y * y));
            }
    });

    RcosFn rc = rcos_fn();
    Vec Yrcos(rc.Y.size()), YIrcos(rc.Y.size());
    for (size_t i = 0; i < rc.Y.size(); ++i) {
        Yrcos[i] = std::sqrt(rc.Y[i]);
        YIrcos[i] = std::sqrt(1.0 - Yrcos[i] * Yrcos[i]);
    }
    const Lut lo0lut{YIrcos, rc.X[0], rc.X[1] - rc.X[0]};
    const Lut hi0lut{Yrcos, rc.X[0], rc.X[1] - rc.X[0]};

    // The single forward transform is parallel over its lines at either level.
    const CMat imdft = mc::fftshift(mc::fft2(im, cfg), cfg);
    sc[0].lodft = CMat(R, C);
    par_for(cfg, C, Sched{}, [&](long lo, long hi) {
        for (size_t i = static_cast<size_t>(lo) * R; i < static_cast<size_t>(hi) * R; ++i)
            sc[0].lodft.d[i] = imdft.d[i] * point_op(lo0lut, sc[0].log_rad.d[i]);
    });

    // Radial lookup tables: the abscissa moves down one octave per scale.
    Vec Xrcos = rc.X;
    for (double& x : Xrcos) x -= 1.0;
    sc[0].hilut = Lut{Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0]};
    for (size_t i = 0; i < Yrcos.size(); ++i) YIrcos[i] = std::fabs(std::sqrt(1.0 - Yrcos[i] * Yrcos[i]));
    const Lut lo1lut{YIrcos, Xrcos[0], Xrcos[1] - Xrcos[0]};
    for (double& x : Xrcos) x -= 1.0;
    sc[1].hilut = Lut{Yrcos, Xrcos[0], Xrcos[1] - Xrcos[0]};

    // Scale 1: central half of the low-pass spectrum and of the coordinates, times the
    // low-pass mask. A copy per element; chunks write whole columns of the three outputs.
    const int lod[2] = {static_cast<int>(std::ceil((R - 0.5) / 2)), static_cast<int>(std::ceil((C - 0.5) / 2))};
    const int loctr[2] = {static_cast<int>(std::ceil((lod[0] + 0.5) / 2)), static_cast<int>(std::ceil((lod[1] + 0.5) / 2))};
    const int r0 = ctr1 - loctr[0], c0 = ctr2 - loctr[1];
    sc[1].log_rad = Mat(lod[0], lod[1]);
    sc[1].angle = Mat(lod[0], lod[1]);
    sc[1].lodft = CMat(lod[0], lod[1]);
    par_for(cfg, lod[1], Sched{}, [&](long lo, long hi) {
        for (int c = static_cast<int>(lo); c < hi; ++c)
            for (int r = 0; r < lod[0]; ++r) {
                const double lr = sc[0].log_rad(r0 + r, c0 + c);
                sc[1].log_rad(r, c) = lr;
                sc[1].angle(r, c) = sc[0].angle(r0 + r, c0 + c);
                sc[1].lodft(r, c) = point_op(lo1lut, lr) * sc[0].lodft(r0 + r, c0 + c);
            }
    });

    // The 13 inverse transforms. Unit 0 is the high-pass residual, units 1-6 the scale-0
    // bands, units 7-12 the scale-1 bands (a quarter of the work each), so the large units
    // are handed out first. Each unit reads the shared spectra and tables and writes one
    // output image of its own: no locks are needed.
    const Lut angle_lut = angle_table(kBands);
    SteerablePyramid pyr;
    auto unit = [&](int u, const Config& c) {
        if (u == 0) {
            CMat hi0dft(R, C);
            par_for(c, C, Sched{}, [&](long lo, long hi) {
                for (size_t i = static_cast<size_t>(lo) * R; i < static_cast<size_t>(hi) * R; ++i)
                    hi0dft.d[i] = imdft.d[i] * point_op(hi0lut, sc[0].log_rad.d[i]);
            });
            pyr.hi0 = mc::ifft2_real(mc::ifftshift(hi0dft, c), c);
            return;
        }
        const int level = (u - 1) / kBands, b = (u - 1) % kBands;
        Lut lut = angle_lut;  // private copy with the origin of this orientation
        lut.origin = angle_lut.origin + kPi * b / kBands;
        pyr.band[level][b] = oriented_band(sc[level], lut, c);
    };
    const int units = 1 + 2 * kBands;
    if (cfg.m4_level == M4Level::Bands) {
        // Whole transforms on different threads, each serial inside. The units are few and
        // unequal, so they are handed out one at a time (dynamic) rather than in blocks.
        Config inner = cfg;
        inner.threads = 1;
        par_for(cfg, units, Sched{SchedKind::Dynamic, 1}, [&](long lo, long hi) {
            for (long u = lo; u < hi; ++u) unit(static_cast<int>(u), inner);
        });
    } else {
        // The 1-D transforms of each unit share all the threads; inside the task graph the
        // units are tasks as well, so a thread never waits for a unit it does not need.
        par_outer(cfg, units, [&](int u) { unit(u, cfg); });
    }
    return pyr;
}
}  // namespace

PI_NOINLINE SteerablePyramid m4_steerable_pyramid(const Mat& im, const Config& cfg) {
    const int R = im.rows, C = im.cols;
    const int ht = 2;
    if (ht > static_cast<int>(std::floor(std::log2(std::min(R, C)))) - 2)
        throw std::runtime_error("image too small for a 2-level steerable pyramid");
    if (cfg.designed()) return steerable_pyramid_designed(im, cfg);
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
